'''
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

 

Example 1:

Input: s = "anagram", t = "nagaram"

Output: true

Example 2:

Input: s = "rat", t = "car"

Output: false

 

Constraints:

1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
'''

# # solution(not optimized)
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         return sorted(s) == sorted(t)
    
# # solution(not optimized)
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         if len(s) != len(t) :
#             return False

#         s_dict = {}
#         t_dict = {}

#         for i in range(len(s)) :
#             s_dict[s[i]] = s.count(s[i])
#             t_dict[t[i]] = t.count(t[i])

#         if s_dict == t_dict :
#             return True
#         else :
#             return False
        
# solution(optimized)
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t) :
            return False

        for ele in set(s) :
            if s.count(ele) != t.count(ele) :
                return False
        
        return True