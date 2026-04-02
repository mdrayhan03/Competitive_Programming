'''
A. String Rotation Game
time limit per test1 second
memory limit per test256 megabytes
Define a block in a string as a contiguous substring of characters of the same type that cannot be extended either to the left or the right. For example, in the string aabcccdaa, there are five blocks:

aa (1
-st to 2
-nd characters)
b (3
-rd character)
ccc (4
-th to 6
-th characters)
d (7
-th character)
aa (8
-th to 9
-th characters).
You are playing a game where you are given a string 𝑠
 of length 𝑛
. You can cyclically rotate∗
 the string however you want. Your score is then calculated as the number of blocks in the final string. Please find the maximum score possible.

∗
Formally, choose an index 1≤𝑖≤𝑛
, and replace the string 𝑠1𝑠2…𝑠𝑛
 with the string 𝑠𝑖+1𝑠𝑖+2…𝑠𝑛𝑠1𝑠2…𝑠𝑖
. For example, the string abcde can be rotated to string deabc by choosing 𝑖=3
.

Input
Each test contains multiple test cases. The first line contains the number of test cases 𝑡
 (1≤𝑡≤500
). The description of the test cases follows.

The first line of each test case contains a single integer 𝑛
 (1≤𝑛≤100
).

The second line of each test case contains the string 𝑠
 of length 𝑛
.

Strings 𝑠
 consist of lowercase Latin characters only.

Output
For each testcase, output a single integer denoting the maximum score you can achieve.

Example
InputCopy
4
4
abcd
4
abbc
4
abba
6
abbccc
OutputCopy
4
4
3
4
Note
In the first test case, score of the original string abcd is 4
. It can be shown that a score greater than 4
 cannot be achieved.

In the second test case, cyclically rotating the string by 2
 positions will give us string bcab. Score of this string is 4
. It can be shown that a score greater than 4
 cannot be achieved.
'''

# solution
def score_counter(s) :
    prev = ''
    cnt = 0
    for ele in s :
        if prev != ele :
            cnt += 1
        prev = ele
    
    return cnt

_n = int(input())
for _ in range(_n) :
    n = int(input())
    s = input()
    score = 0

    for i in range(n) :
        curr_score = score_counter(s[i:]+s[:i])
        if curr_score > score :
            score = curr_score
    
    print(score)