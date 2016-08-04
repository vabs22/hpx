//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async.hpp>
#include <hpx/apply.hpp>
#include <hpx/lcos/detail/barrier_node.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/intrusive_ptr.hpp>

#include <string>
#include <utility>
#include <vector>

typedef hpx::components::managed_component<hpx::lcos::detail::barrier_node> barrier_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::detail::barrier_node, hpx::components::component_barrier)

HPX_REGISTER_ACTION(hpx::lcos::detail::barrier_node::gather_action,
    barrier_node_gather_action);

namespace hpx { namespace lcos { namespace detail {

    barrier_node::barrier_node()
      : count_(0),
        local_barrier_(0)
    {
        HPX_ASSERT(false);
    }

    barrier_node::barrier_node(std::string base_name, std::size_t num, std::size_t rank)
      : count_(0),
        base_name_(base_name),
        rank_(rank),
        num_(num),
        arity_(std::stol(get_config_entry("hpx.lcos.collectives.arity", 5))),
        cut_off_(std::stol(get_config_entry("hpx.lcos.collectives.cut_off", 16))),
        local_barrier_(num)
    {
        if (num_ >= cut_off_)
        {
            std::vector<std::size_t> ids;
            ids.reserve(children_.size());

            for (std::size_t i = 1; i <= arity_; ++i)
            {
                std::size_t id = (arity_ * rank_) + i;
                if (id >= num) break;
                ids.push_back(id);
            }

            children_ = hpx::util::unwrapped(
                hpx::find_from_basename(base_name_, ids));

            return;
        }

        if (rank_ != 0)
        {
            children_.push_back(hpx::find_from_basename(base_name_, 0).get());
        }
    }

    hpx::future<void> barrier_node::wait(bool async)
    {
        if (num_ < cut_off_)
        {
            if (rank_ != 0)
            {
                HPX_ASSERT(children_.size() == 1);
                hpx::lcos::base_lco::set_event_action action;
                return hpx::async(action, children_[0]);
            }
            else
            {
                if (async)
                {
                    boost::intrusive_ptr<barrier_node> this_(this);
                    return hpx::async(&barrier_node::set_event, this_);
                }
                set_event();
                return hpx::make_ready_future();
            }
        }

        // We only set our flag once all children entered the barrier and have
        // the flag set.
        std::vector<hpx::future<void> > futures;
        futures.reserve(children_.size());
        for(hpx::id_type& id : children_)
        {
            barrier_node::gather_action action;
            futures.push_back(hpx::async(action, id));
        }
        // Keep ourself alive, we only need to do additionally refcount if the
        // barrier is asynchronous
        if(async)
        {
            boost::intrusive_ptr<barrier_node> this_(this);
            return do_wait(this_, std::move(futures));
        }
        return do_wait(this, std::move(futures));
    }

    template <typename This>
    hpx::future<void> barrier_node::do_wait(This this_,
        std::vector<hpx::future<void> > futures)
    {
        hpx::future<void> res = broadcast_promise_.get_future();
        return
            hpx::when_all(futures).then(hpx::launch::sync,
                [this_](hpx::future<std::vector<hpx::future<void>>> f)
                {
                    // trigger possible errors
                    f.get();

                    // Once the children have entered the barrier, we can set our flag.
                    this_->gather_promise_.set_value();

                    // We now need to reset the flag of our children to mark the barrier
                    // done...
                    std::vector<hpx::future<void> > futures;
                    futures.reserve(this_->children_.size());
                    for(hpx::id_type& id : this_->children_)
                    {
                        base_lco::set_event_action action;
                        futures.push_back(hpx::async(action, id));
                    }

                    return hpx::when_all(futures);
                }
            ).then(hpx::launch::sync,
                hpx::util::bind(
                    hpx::util::one_shot(
                        [this_](hpx::future<std::vector<hpx::future<void>>> f,
                            hpx::future<void> result)
                        {
                            // trigger possible errors
                            f.get();
                            // The root process has no children that sets its flag...
                            if (this_->rank_ == 0)
                            {
                                this_->set_event();
                            }
                            return result;
                        }
                    )
                  , hpx::util::placeholders::_1
                  , std::move(res)
                )
            );
    }

    template hpx::future<void>
        barrier_node::do_wait(
            boost::intrusive_ptr<barrier_node>, std::vector<hpx::future<void> >);
    template hpx::future<void>
        barrier_node::do_wait(
            barrier_node*, std::vector<hpx::future<void> >);

    hpx::future<void> barrier_node::gather()
    {
        return gather_promise_.get_future();
    }

    void barrier_node::set_event()
    {
        if (num_ < cut_off_)
        {
            local_barrier_.wait();
            return;
        }

        // the barrier needs to wait for its flag to be set.
        // The leave nodes set it first due to having no children.
        // reset promise for another round...
        gather_promise_ = hpx::lcos::local::promise<void>();
        hpx::lcos::local::promise<void> tmp;
        tmp.swap(broadcast_promise_);
        tmp.set_value();
    }
}}}
