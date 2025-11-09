# Exercise 1: Dependency Detective

**Module:** 02 - Loop Dependencies  
**Difficulty:** Intermediate  
**Time:** 15-20 minutes

## Learning Objectives
- Identify loop-carried dependencies
- Classify dependency types (RAW, WAR, WAW, reduction)
- Determine which loops can be vectorized

## Problem

Analyze the following 8 loops and for each:
1. Identify if there's a loop-carried dependency
2. If yes, classify the type (true/RAW, anti/WAR, output/WAW, reduction)
3. Determine: Can it be vectorized? (Yes/No/Maybe)
4. If "Maybe", explain what conditions must be met

```c
// Loop A
for (int i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
}

// Loop B
for (int i = 1; i < n; i++) {
    a[i] = a[i-1] + b[i];
}

// Loop C
for (int i = 0; i < n; i++) {
    sum += a[i];
}

// Loop D
for (int i = 0; i < n-1; i++) {
    a[i] = a[i+1] * 2.0;
}

// Loop E
for (int i = 0; i < n; i++) {
    a[indices[i]] = b[i];
}

// Loop F
for (int i = 2; i < n; i++) {
    a[i] = a[i-2] + 1.0;
}

// Loop G
for (int i = 0; i < n; i++) {
    if (a[i] > max) max = a[i];
}

// Loop H
for (int i = 0; i < n; i++) {
    a[i] = b[i];
    b[i] = c[i];
}
```

## Answer Format

Fill in this table:

| Loop | Dependency? | Type | Vectorizable? | Notes |
|------|-------------|------|---------------|-------|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |
| E | | | | |
| F | | | | |
| G | | | | |
| H | | | | |

## Hints

- Remember: RAW (Read-After-Write) is a true dependency
- WAR (Write-After-Read) might be vectorizable depending on access pattern
- Reductions are special - can be vectorized despite dependencies
- Distance matters! Distance-2 dependencies might allow vectorization

## Solution

See `solutions/exercise1_solution.md`
