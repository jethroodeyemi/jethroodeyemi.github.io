---
title: 'Algorithm — Reverse Only Letters: A Python Solution'
date: 2024-03-14
permalink: /posts/2024/03/algo-reverse-only-letters/
tags:
  - algorithm
  - python
  - string
  - leetcode
---

In many programming interviews, candidates encounter challenges that test their ability to manipulate strings efficiently. One such problem involves reversing a string with a twist: only the letters should be reversed, while non-letter characters remain in their original positions. In this blog post, we'll explore this problem and present an optimized Python solution.

## Problem Statement

Given a string s, we need to reverse it according to the following rules:
- All characters that are not English letters remain in the same position.
- All English letters (both lowercase and uppercase) should be reversed.

## Initial Approach

Initially, one might consider creating a new string by iterating through the input string and reversing letters while keeping non-letter characters unchanged. However, this approach can be inefficient, especially for longer strings.

## Optimized Solution

To optimize the solution, we can utilize a two-pointer approach. We'll use pointers to traverse the string from both ends, swapping letters when both pointers are positioned on letters. Here's the Python implementation:

```python
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        s = list(s)
        l, r = 0, len(s) - 1
        while l < r:
            if not s[l].isalpha():
                l += 1
            elif not s[r].isalpha():
                r -= 1
            else:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
        return ''.join(s)
```

## Explanation

- We start by converting the input string into a list to allow for in-place manipulation.
- Using two pointers (`l` and `r`), we iterate through the string from both ends towards the center.
- At each step, we check if both characters at the pointers are letters. If not, we increment/decrement the pointers accordingly.
- When both pointers point to letters, we swap them, effectively reversing the letters.
- We continue this process until the pointers meet or cross each other.

## Time and Space Complexity

- Time Complexity: The time complexity of this solution is O(n), where n is the length of the input string. This is because we iterate through the string only once.
- Space Complexity: The space complexity is also O(n) due to the conversion of the string into a list. However, the algorithm operates in place without using any extra data structures except for the input string.

## Conclusion

In this blog post, we explored an optimized solution to the problem of reversing a string with specific rules. By utilizing a two-pointer approach, we achieved a more efficient solution in terms of time and space complexity. This solution demonstrates the importance of algorithmic optimization in solving programming challenges effectively.

## References

- LeetCode Problem: [917. Reverse Only Letters](https://leetcode.com/problems/reverse-only-letters/)

Thank you for reading! I hope you found this blog post insightful and informative. If you have any questions or suggestions, feel free to leave a comment below. Don't forget to like and share if you found this content helpful. Happy coding!