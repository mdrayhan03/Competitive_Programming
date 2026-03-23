'''
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true
Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
 

Constraints:

0 <= s.length <= 100
0 <= t.length <= 104
s and t consist only of lowercase English letters.
'''

# solution

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        s_list = []

        def is_sorted(nums):
            return all(nums[i] <= nums[i+1] for i in range(len(nums) - 1))

        for ele in s:
            if ele in t:
                index = -1
                for i in range(t.count(ele)) :
                    index = t.index(ele, index + 1)
                    if len(s_list) == 0 :
                        s_list.append(index)
                        break
                    elif index > s_list[-1] :
                        s_list.append(index)
                        break

            else :
                return False
        
        if len(s_list) != len(s) :
            return False
            
        return is_sorted(s_list)

print(Solution().isSubsequence(s = "", t = ""))