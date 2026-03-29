'''
A. Array Coloring
time limit per test1 second
memory limit per test256 megabytes

You have 𝑛
 cards arranged in a row. The 𝑖
-th card has the integer 𝑎𝑖
 written on it. All integers are distinct.

You must color each card either red
 or blue
 such that the following conditions are satisfied:

Any two adjacent cards in the row have different colors.
If you rearrange the cards so that the numbers on them are in increasing order, any two adjacent cards in the new row must also have different colors.
Determine if such a coloring exists.

Input
Each test contains multiple test cases. The first line contains the number of test cases 𝑡
 (1≤𝑡≤200
). The description of the test cases follows.

The first line of each test case contains a single integer 𝑛
 (2≤𝑛≤100
) — the length of the array.

The second line of each test case contains 𝑛
 integers 𝑎1,𝑎2,…,𝑎𝑛
 (1≤𝑎𝑖≤𝑛
).

It is guaranteed that all elements of 𝑎
 are distinct.

Output
For each test case, output "YES" if you can color the cards so that the conditions are satisfied, and "NO" otherwise.

You can output the answer in any case (upper or lower). For example, the strings "yEs", "yes", "Yes", and "YES" will be recognized as positive responses.

Example
InputCopy
4
4
2 3 4 1
3
2 3 1
5
3 4 1 2 5
5
3 1 4 2 5
OutputCopy
YES
NO
YES
NO
Note
In the first example, the cards are colored as 𝑎=[2,3,4,1]
. After sorting, the cards become [1,2,3,4]
. Both sequences satisfy the condition, so the answer is YES.

In the second example, no valid coloring exists. For instance, if we color the cards as 𝑎=[2,3,1]
, the sorted sequence becomes [1,2,3]
. Here, the adjacent elements 1
 and 2
 have the same color, so the condition is not satisfied.

In the third example, a possible coloring is 𝑎=[3,4,1,2,5]
. After sorting, the cards become [1,2,3,4,5]
.
'''

# solution
t = int(input())

for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))

    result = {num: "BLUE" if i % 2 != 0 else "RED" for i, num in enumerate(a)}

    a.sort()

    # 3. Check the sorted neighbors
    for i in range(1, n):
        if result[a[i]] == result[a[i-1]]:
            print("NO")
            break
    else:
        print("YES")