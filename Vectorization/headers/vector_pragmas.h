

/* ────────────────────────────────────────────────────────────────────────── +
 │                                                                            │
 │ This file is part of the exercises for the Lectures on                     │
 │   "Advanced High Performance Computing"                                    │
 │ given at                                                                   │
 │ Scientific & Data-Intensive Computing @ University of Trieste              │
 │                                                                            │
 │ contact: luca.tornatore@inaf.it                                            │
 │                                                                            │
 │     This is free software; you can redistribute it and/or modify           │
 │     it under the terms of the GNU General Public License as published by   │
 │     the Free Software Foundation; either version 3 of the License, or      │
 │     (at your option) any later version.                                    │
 │     This code is distributed in the hope that it will be useful,           │
 │     but WITHOUT ANY WARRANTY; without even the implied warranty of         │
 │     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          │
 │     GNU General Public License for more details.                           │
 │                                                                            │
 │     You should have received a copy of the GNU General Public License      │
 │     along with this program.  If not, see <http://www.gnu.org/licenses/>   │
 │                                                                            │
 + ────────────────────────────────────────────────────────────────────────── */


/* @ ······················································· @
   :  Here we collect some macros to express vectorization   :
   :  Pragmas indpendently on the compiler used.             :
   :  This is intended to have only didactial purposes and   :
   :  by no means conveys an exhaustive list and/or          :
   :  combination, nor the maximum tuning for every compiler :
   @ ······················································· @
 */


// the two following macros are useful only to
// produce pragma strings in the source files
//
#define STRINGIFY(X) #X
#define _DO_PRAGMA(x) _Pragma (#x)
//

/* ············································································
 *  
 *  GCC
 */

#if defined(__GNUC__) && !defined(__clang__)
#pragma message "using GCC"
#define IVDEP              _Pragma("GCC ivdep")
#define LOOP_VECTORIZE     _Pragma("GVV ivdep")
#define LOOP_VECTOR_LENGTH(N)
#define LOOP_UNROLL        _Pragma("GCC unroll")
#define LOOP_UNROLL_N(N)   _DO_PRAGMA(GCC unroll N)

/* ············································································
 *
 *  CLANG
 */

#elif defined(__clang__)
#define IVDEP                 _Pragma("clang ivdep")
#define LOOP_VECTORIZE        _DO_PRAGMA(clang loop vectorize( enable ))
#define LOOP_VECTOR_LENGTH(N) _DO_PRAGMA(clang vectorize_width( N ))
#define LOOP_UNROLL           _DO_PRAGMA(clang loop interleave( enable ))
#define LOOP_UNROLL_N(N)      _DO_PRAGMA(clang loop interleave_count( N ))

/* ············································································
 *
 *  INTEL COMPILER
 */
 
#elif defined(__ICC)
#define IVDEP            _Pragma("ivdep")
#define LOOP_VECTORIZE   _Pragma("vector")
#define LOOP_UNROLL      _Pragma("unroll")
#define LOOP_UNROLL_N(N) _DO_PRAGMA(unroll N)
#define VECTOR_ALWAYS    _Pragma("vector always")
#define VECTOR_ALIGNED   _Pragma("vector aligned")
#define VECTOR_UNALIGNED _Pragma("vector unaligned")



/* ············································································
 *
 *  ARM COMPILER
 */
#elif defined(__CC_ARM)

// TBD, sorry

#endif


#if !defined(__ICC)
#define VECTOR_ALWAYS      _Pragma("message \"vector always not defined\"")
#define VECTOR_ALIGNED     _Pragma("message \"vector aligned not defined\"")
#define VECTOR_UNALIGNED   _Pragma("message \"vector unaligned not defined\"")
#endif
