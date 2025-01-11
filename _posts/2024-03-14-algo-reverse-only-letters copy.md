---
title: "Algorithm — Generate Parentheses: Python and C++ Solutions"
date: 2025-01-11
permalink: /posts/2025/01/algo-generate-parentheses/
tags:
  - algorithm
  - python
  - cpp
  - leetcode
---

In this blog post, we'll explore LeetCode's "Generate Parentheses" problem and present two clean backtracking solutions—one in Python and one in C++. This classic problem is an excellent demonstration of how to use recursion to systematically explore all valid possibilities.

## Problem Statement

Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example 1:**

```text
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Example 2:**

```text
Input: n = 1
Output: ["()"]
```

**Constraints:**

- (1 <= n <= 8)

## Key Insight

A "well-formed" parentheses string adheres to the following rules:

- It must contain exactly `n` opening parentheses '(' and `n` closing parentheses ')'.
- At any point in the string, we cannot have used more `')'` characters than `'('` characters (otherwise it’s invalid).

To generate all valid parentheses, we leverage **backtracking**:

1. Maintain two counters:
   - `l` (the number of opening parentheses used so far)
   - `r` (the number of closing parentheses used so far)
2. At each step:
   - We can add `'('` if `l < n`.
   - We can add `')'` if `r < l`.

By following these constraints, we only build valid combinations and avoid invalid partial strings early.

## Python Solution

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []

        def backtrack(s, l, r):
            # If the current string is of length 2*n, we've used all parentheses
            if len(s) == n * 2:
                result.append(s)
                return

            # If we can still place an opening parenthesis, place it
            if l < n:
                backtrack(s + '(', l + 1, r)

            # If we have more '(' placed than ')', we can place another ')'
            if r < l:
                backtrack(s + ')', l, r + 1)

        backtrack("", 0, 0)
        return result
```

### Explanation

1. We create an empty list `result` to store the valid parentheses combinations.
2. The helper function `backtrack`:
   - Appends the current combination `s` to `result` once it reaches a length of `2 * n`.
   - Tries adding `'('` if we have not yet used all `n` opening parentheses.
   - Tries adding `')'` if the number of `')'` used so far is less than the number of `'('` used.
3. **Note**: The condition `r < l` ensures that we do not place more closing parentheses than opening parentheses, keeping our strings valid.

### Time and Space Complexity

- **Time Complexity**: \( O(C_n) \), where \( C_n \) is the \( n^\text{th} \) Catalan number, \( \frac{1}{n+1}\binom{2n}{n} \). In simpler terms, the backtracking visits every valid sequence exactly once.
- **Space Complexity**: \( O(n) \) for the recursion call stack in generating each valid sequence (not counting the output storage).

## C++ Solution

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        backtrack(result, "", 0, 0, n);
        return result;
    }

private:
    void backtrack(vector<string>& result, string s, int l, int r, int n) {
        // If the current string is of length 2*n, we've used all parentheses
        if (s.length() == 2 * n) {
            result.push_back(s);
            return;
        }

        // If we can still place an opening parenthesis, place it
        if (l < n) {
            backtrack(result, s + '(', l + 1, r, n);
        }

        // If we have more '(' placed than ')', we can place another ')'
        if (r < l) {
            backtrack(result, s + ')', l, r + 1, n);
        }
    }
};
```

### Explanation

- We follow the same logic as in the Python solution.
- The helper function `backtrack` takes additional parameters for `l` and `r` which track how many `'('` and `')'` have been used.
- We add the formed string to `result` once it reaches the appropriate length (`2 * n`).

### Time and Space Complexity

- **Time Complexity**: \( O(C_n) \) (the \( n^\text{th} \) Catalan number).
- **Space Complexity**: \( O(n) \) for the depth of the recursion stack.

## Conclusion

The "Generate Parentheses" problem is a quintessential example of backtracking. By carefully constraining the addition of `'('` and `')'`, we ensure that we only build valid combinations without redundant or invalid branches. This is a powerful technique that appears in many similar string-generation and combinatorial problems.

If you found this post helpful, consider sharing it or leaving a comment below! Happy coding!

## References

- LeetCode Problem: [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
