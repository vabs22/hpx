//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_STATIC_LINKING)
#include <hpx/hpx_init_impl.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
// Default implementation of main() if all the user provides is
// hpx_startup::user_main.
//
// This has to be in a separate translation unit to ensure the linker can pick
// or ignore this function, depending on whether the main executable defines
// this symbol or not.
int main(int argc, char** argv)
{
    return hpx::init(argc, argv);
}

