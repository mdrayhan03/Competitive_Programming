'''
Given a 32-bit integer num, return a string representing its hexadecimal representation. For negative integers, two’s complement method is used.

All the letters in the answer string should be lowercase characters, and there should not be any leading zeros in the answer except for the zero itself.

Note: You are not allowed to use any built-in library method to directly solve this problem.

 

Example 1:

Input: num = 26
Output: "1a"
Example 2:

Input: num = -1
Output: "ffffffff"
 

Constraints:

-231 <= num <= 231 - 1
'''

# solution
from typing import List

class Solution:
    def toHex(self, num: int) -> str:
        if num == 0 : return "0"

        hex_map = "0123456789abcdef"
        is_neg = False
        
        if num < 0 :
            is_neg = True

        num = abs(num)

        def twos_complement(nums: List[int]) -> List[int]:
            nums = nums + [0] * (8 - len(nums))

            reminder = 1
            for i in range(len(nums)) : 
                digit = ((15 - nums[i]) + reminder)
                nums[i] = digit % 16

                if digit // 16 > 0 :
                    reminder = 1
                else :
                    reminder = 0
            
            return nums
        
        hex_list = []

        while num > 0 :
            hex_list.append(num % 16)
            num = num // 16

        if is_neg :
            hex_list = twos_complement(hex_list)

        for i in range(len(hex_list)) :
            hex_list[i] = hex_map[hex_list[i]]

        hex_list = hex_list[::-1]
        return ''.join(hex_list)