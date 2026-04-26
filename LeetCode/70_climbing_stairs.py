'''
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
 

Constraints:

1 <= n <= 45
'''

# solution
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        
        # 'first' represents n-2, 'second' represents n-1
        first, second = 1, 2
        
        for i in range(3, n + 1):
            # The current step is the sum of the previous two
            current = first + second
            # Update pointers for the next iteration
            first = second
            second = current
            
        return second
    
# solution
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {1: 1, 2: 2}
        
        def calculate(n):
            if n in memo:
                return memo[n]
            
            memo[n] = calculate(n - 1) + calculate(n - 2)
            return memo[n]
            
        return calculate(n)