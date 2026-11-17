---
name: doxygen-style
description: Use this skill whenever writing or editing Doxygen documentation comments in Taskflow's C++ headers (taskflow/**/*.hpp). Triggers include adding @brief/@param/@return/@code documentation to a class or function, reviewing existing doc comments for style consistency, or being asked to "doxygen-ify" / "document" / "add doc comments to" a header file. Encodes the project's required comment block format, tag ordering, and example style so generated docs match the existing codebase exactly.
---

# Doxygen Style

This skill describes the exact Doxygen documentation conventions used across
`taskflow/**/*.hpp`. Follow it precisely — formatting deviations (stray `///`
comments, wrong tag order, missing examples) are the most common review
feedback on doc passes in this repo.

## Comment block syntax

- Use only `/** ... */` block comments for Doxygen documentation.
- **Never** use `///` line comments (including trailing `///<` field
  comments) and **never** use multi-star banner comments (`/********/`).
- Inside the block, do **not** prefix each line with `* `. Lines start flush
  left (or at the member's indentation level), with no leading asterisk.
- Use markdown backtick syntax (`` `x` ``) for inline code/identifiers instead
  of `@c x` — Doxygen's `@c` command doesn't always render reliably in this
  codebase's setup. Reserve `@c` only for cases already in surrounding code
  that you're not otherwise touching.

Wrong:
```cpp
/**
 * @brief does something
 */
///< trailing field comment
/********************************/
```

Right:
```cpp
/**
@brief does something
*/
```

## Tag ordering within a block

Every documented function, method, or class follows this fixed order:

1. `@brief` — one line, imperative/descriptive, no trailing period required.
2. `@class` (classes only) — immediately before or after `@brief`.
3. Signature-related tags, in this order: `@tparam` (one per template
   parameter, in declaration order), then `@param` (one per parameter, in
   declaration order), then `@return`.
4. Explanation prose — one or more plain paragraphs describing behavior,
   algorithmic detail, or rationale. This comes **after** the signature tags,
   not before.
5. `@code{.cpp} ... @endcode` — a minimal, realistic usage example.
6. `@note`, `@warning` — caveats, thread-safety notes, lifetime rules. These
   always come **last**, after the `@code` example. Prefer `@note` by default;
   reserve `@warning` for cases describing actual undefined behavior or
   correctness/safety hazards. Avoid `@attention` — use `@note` instead.

Skip any tag that doesn't apply (e.g., a `void`-returning function has no
`@return`; a non-template function has no `@tparam`). Never reorder the tags
that are present.

Template skeleton:
```cpp
/**
@brief <one-line summary>

@tparam T <template parameter description>
@param x <parameter description>
@return <return value description>

<Explanation paragraph(s): what it does, how, and why, in plain prose.>

@code{.cpp}
<minimal realistic usage example>
@endcode

@note <caveat, e.g. thread-safety, lifetime conditions; use @warning instead for actual UB/safety hazards>
*/
```

## Examples from the codebase

Free function:
```cpp
/**
@brief rounds the given 64-bit unsigned integer to the nearest power of 2

@tparam T 64-bit unsigned integral type
@param x the number to round up
@return the smallest power of 2 that is greater than or equal to @c x

This overload participates in overload resolution only when @c T is an
8-byte unsigned integral type. It repeatedly fills in the lower bits of
<tt>x - 1</tt> until all bits below the highest set bit are 1, then adds 1
to obtain the next power of 2.

@code{.cpp}
tf::next_pow2(uint64_t{17});  // returns 32
tf::next_pow2(uint64_t{32});  // returns 32
@endcode
*/
template <typename T>
requires (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8)
constexpr T next_pow2(T x) { ... }
```

Class member with multiple caveats:
```cpp
/**
@brief checks if the given number is a power of 2

@tparam T integral type of the input
@param x The integer to check.
@return `true` if `x` is a power of 2, otherwise `false`.

This function determines if the given integer is a power of 2 by testing
that exactly one bit is set, i.e., <tt>x & (x - 1) == 0</tt>, while also
excluding zero.

@code{.cpp}
tf::is_pow2(8);   // true
tf::is_pow2(10);  // false
@endcode

@note This function is constexpr and can be evaluated at compile time.
*/
```

Class-level doc (uses `@class`, no signature tags, `@note` last since
there's no separate member doc to attach it to):
```cpp
/**
@class Xorshift

@brief class to create a fast xorshift-based pseudo-random number generator

@tparam T unsigned integral type used as the internal state (supported uint32_t and uint64_t)

This class implements a lightweight xorshift pseudo-random number generator
suitable for performance-critical paths such as schedulers, work-stealing
victim selection, and randomized backoff.

@note
The internal state must be seeded with a non-zero value.
This class is not thread-safe. Each thread should maintain its own instance.
*/
```

## Scope and rendering notes

- Only files listed in `doxygen/Doxyfile`'s `INPUT` are rendered into the
  public docs. Check that list before assuming a header is user-facing.
- `EXTRACT_ALL = NO` and `EXTRACT_PRIVATE = NO` in `doxygen/Doxyfile` mean
  `@private`-tagged entities and private class members are never rendered.
  Apply the syntax cleanup (no `///`, no banner stars) to these for
  consistency, but don't bother adding `@code` examples — they aren't
  user-facing.
- Plain `//` comments on genuinely internal/private members (e.g. a private
  nested helper struct never reachable from the public API) should be left
  as-is; they aren't Doxygen comments and don't need conversion.

## Verification checklist after editing a header

1. `grep -n "^\s*///" <file>` — must return nothing.
2. `grep -n "/\*\*\*\|\*\*\*\*" <file>` — must return nothing (no banner-star
   comments).
3. `g++ -std=c++20 -I. -fsyntax-only <file>` — must compile clean (ignore the
   benign `#pragma once in main file` warning).
4. Where feasible, compile and run a small standalone program that exercises
   the documented `@code` examples, to confirm they're not just plausible
   but actually correct.
