
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
   :  We attempt here to uniquely define "our" vector types  :
   :  not using intrinsics but the custom vector types.      :
   :  We exploit the fact that the                           :
   :      __attribute__((vector size (N)))                   :
   :  formalism is widely adopted.                           :
   :                                                         :
   :  This is intended to have only didactical purposes and  :
   :  by no means conveys an exhaustive list and/or          :
   :  combination, nor the maximum tuning for every compiler :
   @ ······················································· @
 */



// --------------------------------------------------------
//  discover the vector size
// --------------------------------------------------------

#if !defined(VSIZE)

#if !defined(_IMMINTRIN_H_INCLUDED) && !defined(__IMMINTRIN_H) && !defined()
#include <immintrin.h>
#endif

#if defined(__AVX512__)

#warning "found AVX512"
#define VSIZE (sizeof(__m512))

#define DVSIZE (VSIZE / sizeof(double))
#define FVSIZE (VSIZE / sizeof(float))
#define IVSIZE (VSIZE / sizeof(int))

#elif defined ( __AVX__ ) || defined ( __AVX2__ )

#warning "found AVX/AVX2"
#define VSIZE (sizeof(__m256))

#elif defined ( __SSE4__ ) || defined ( __SSE3__ )

#warning "found SSE >= 3"
#define VSIZE (sizeof(__m128))

#else

#define VSIZE (sizeof(double))
#endif


// --------------------------------------------------------
//  VSIZE has been either given or discovered
// --------------------------------------------------------

#if ( ((VSIZE-1)&VSIZE) > 0 )
#error "the defined vector size is not a power of 2"
#endif

#if (VSIZE<=sizeof(double))

#define NO_VECTOR
#warning "no vector capability found"
typedef double dvector_t;
typedef float fvector_t;
typedef int ivector_t;

#define DVSIZE 1
#define FVSIZE 1
#define IVSIZE 1

#else


#define DVSIZE (VSIZE / sizeof(double))
#define FVSIZE (VSIZE / sizeof(float))
#define IVSIZE (VSIZE / sizeof(int))

typedef double dvector_t __attribute__((vector_size (VSIZE)));
typedef float  fvector_t __attribute__((vector_size (VSIZE)));
typedef int    ivector_t __attribute__((vector_size (VSIZE)));


#endif
