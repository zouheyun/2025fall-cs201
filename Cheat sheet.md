Cheat sheet           

目录：

```python
#  1.heapq语法  2.切片语法  3.deque语法  4.反转链表  5.类的定义  6.循环链表  8.BFS语法  9.sys语法  10.bisect语法  11.双堆法&大小堆法  12.回溯语法  13.动态规划问题  14.中序转后序  15.归并排序  16.哈希表  18.递归+回溯  19集合语法  20.类型提示  21.字典语法  22.全局变量声明  23.字符串判断方法  24.完全二叉树  25.搜索树  26.查找元素  27.tire树  28.defaultdict语法  29.邻接表  30.并查集类  31.Vertex类和Graph类  32.遍历字母  33.Dijkstra算法  34.字典推导式  35.无穷大  36.欧拉筛  37.判断入度和出度
```

1.heapq语法

```python
import heapq
heapq.heappush(heap, item)  #添加元素    o(logN)
heapq.heappop(heap) #弹出最小元素，就是堆顶元素list[0]     o(logN)
heapq.heapify(x) #将普通列表转换为堆     o(N)
#空列表可以直接视为空堆
```

2.切片语法

```python
A[n:m] #包含n，不包含m，和range一样
A[:] #浅拷贝，元素为列表的时候绝对不能用
A[-3:-1] #包含倒数第三个元素，不包含倒数第一个元素
```

3.deque语法

```python
from collections import deque
```

| 操作类别       | 方法 (语法)                 | 功能描述与时间复杂度                                  |
| :------------- | :-------------------------- | :---------------------------------------------------- |
| **初始化**     | `deque(iterable, maxlen=N)` | 创建一个deque，可从可迭代对象初始化，可设置最大长度。 |
| **添加元素**   | `append(element)`           | 在右端添加元素。**O(1)**                              |
|                | `appendleft(element)`       | 在左端添加元素。**O(1)**                              |
|                | `extend(iterable)`          | 在右端合并序列。O(k)                                  |
|                | `extendleft(iterable)`      | 在左端合并序列（元素顺序相反）。O(k)                  |
| **移除元素**   | `pop()`                     | 移除并返回右端元素。**O(1)**                          |
|                | `popleft()`                 | 移除并返回左端元素。**O(1)**                          |
|                | `remove(value)`             | 移除首个匹配的元素。O(n)                              |
|                | `clear()`                   | 清空所有元素。O(1)                                    |
| **访问与检查** | `d[index]`                  | 按索引访问。**O(1)**                                  |
|                | `len(d)`                    | 获取长度。**O(1)**                                    |
|                | `count(value)`              | 统计元素出现次数。O(n)                                |
| **特殊操作**   | `rotate(n)`                 | 高效地将序列循环移动n步。O(k)                         |

记住它的设计目标：**为序列两端的高效（O(1)）增删操作而生**。

4.反转链表

```python
        #反转链表
        prev = None
        curr = head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
```

5.类的定义

定义里面必须使用self，第一个方法必须是init，其他属性输入必要时使用None默认输入值。

```python
class ListNode: #双向链表
    def __init__(self, val, prev = None, next = None):
        self.val = val
        self.prev = prev
        self.next = next
    #或者 self.children = children if children is not None else []
    #可以直接输入children参数
	#parent = Node(1, children=[child1, child2]) 
```

属性定义之后可以直接在类之外的地方使用

6.循环链表

```python
       prev_node = head
    # 依次创建 2 到 n 的节点
        for i in range(2, n + 1):
            new_node = ChildNode(i)
            prev_node.next = new_node
            prev_node = new_node
        # 闭合环形结构：末尾指向头
        prev_node.next = head
```

例：josephus问题

7.快慢指针：用于寻找链表中间某处的值

8.BFS语法

```python
# 1. 核心工具：先进先出队列
queue = deque([(0, 0, 0)]) 
while queue:
    # 2. 体现“广度”的关键动作：从最前面弹出（先来的先处理）
    r, c, steps = queue.popleft() 
    # 3. 体现“扩散”：一口气把四周还没走的邻居全塞进队尾
    for dr, dc in directions:
        if 合法:
            queue.append((nr, nc, steps + 1))     
#例题：逃离紫罗兰监狱（有条件的BFS：一定要在queue和visit中存储状态）
#给定grid, R, C, K
    sr, sc = -1, -1
    for r in range(R):
        row = next(tokens)
        grid.append(row)
        if 'S' in row:
            sr, sc = r, row.index('S')
    queue = deque([(sr, sc, 0, 0)])  # (row, col, used_k, steps)
    visited = {(sr, sc, 0)}
    while queue:
        r, c, k, steps = queue.popleft()
        if grid[r][c] == 'E':
            print(steps)
            return
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                target = grid[nr][nc]
                nk = k + 1 if target == '#' else k
                if nk <= K and (nr, nc, nk) not in visited:
                    visited.add((nr, nc, nk))
                    queue.append((nr, nc, nk, steps + 1))
    print(-1)
if __name__ == "__main__":
    solve()
    #DFS和BFS的选取：如果是求排列组合、遍历所有解这些，就要用深度优先，求最短路径就用广度优先
    #BFS的核心是while 直到终点，DFS的核心是递归，直到终点
    #BFS回溯：必须引入一个parents节点，因为不想DFS那样能够直接pop撤销操作！！！
    #记住：只要bfs遍历的时候要求某个遍历过程的量，切记要引入一个新的字典来储存。切记切记！！！！
```

9.sys语法

```python
import sys
#情况1：一个一个操作
# 1. 核心操作：把整个输入读入，切碎，变成一个“叫号机”
tokens = iter(sys.stdin.read().split()) #split返回一个字符串列表
# 情况 A：给定数量 N
try:
    line1 = next(tokens)
    n = int(line1)
    for _ in range(n):
        x = next(tokens)
        # 处理业务逻辑
except StopIteration:
    pass
# 情况 B：不给定数量，读到 EOF 为止
tokens = iter(sys.stdin.read().split())
try:
    while True:
        x = next(tokens)
        y = next(tokens)
        # 只要还能叫到号，就一直处理
except StopIteration:
    # 叫不到号了，说明读完了，优雅退出
    pass
#情况2：一排一排操作（用于每一排的数据没有固定格式的时候很方便）
#法1：每次读一行
grid = []
for _ in range(R):
    line = sys.stdin.readline().strip() # .strip() 非常重要，用来去掉末尾的换行符 \n，返回一个字符串，而不是用split返回一个列表
    grid.append(line)
#法2：先全部读完，再遍历
grid = []
for line in sys.stdin: # 它会自动从当前位置读到文件结束
    clean_line = line.strip()
    if not clean_line: continue # 跳过可能的空行
    grid.append(clean_line)
```

10.bisect语法

```python
import bisect 
a = [1, 2, 2, 2, 4]
# 查找 2 的位置
idx_l = bisect.bisect_left(a, 2)   # 结果：1 (第一个 2 的索引)
idx_r = bisect.bisect_right(a, 2)  # 结果：4 (最后一个 2 之后的索引)
#插入
bisect.insort_left(a, x)   #找到 bisect_left 的位置并插入 x
bisect.insort_right(a, x)  #找到 bisect_left 的位置并插入 x
#insort 是 bisect_right + insert的缩写
#二分查找运用场景：维护动态有序列表
#二分查找维护与heapq维护的比较：
```

| 操作             | 有序列表 (`bisect`)         | 堆 (`heapq`)                                  |
| :--------------- | :-------------------------- | :-------------------------------------------- |
| **添加元素**     | $O(N)$ (插入慢，找位置快)   | $O(\log N)$ (极快)                            |
| **查询中位数**   | **$O(1)$** (数组索引直接取) | $O(1)$ (前提是你有两个堆，否则降至O(N log N)) |
| **删除最早元素** | $O(N)$ (简单直接)           | **极其痛苦** (不支持按顺序删)                 |
| **实现难度**     | 简单（学徒级）              | 复杂（大魔导师级）                            |

11.双堆操作&大小堆操作

```python
#一般指用deque+bisect，用于既要频繁进出又要维护有序的问题
#说白了：找中位数问题，和中位数有关的动态问题
#例：
```

12.回溯语法

```python
#例题：输出一个栈的所有可能pop组合
class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        answer = []
        path = []
        def backtrack(current_nums):
            # 基底：当没有剩余数字可选时，说明 path 已满
            if len(current_nums) == 0:
                # 关键点：存入副本 path[:]，这里使用浅拷贝的前提是所有元素是int类型，不存在浅拷贝问题，否则需要deepcopy
                answer.append(path[:])
                return
            for i in range(len(current_nums)):
                # 选择一个“粒子”放入路径
                path.append(current_nums[i])
                # 这里的 current_nums[:i] + current_nums[i+1:] 相当于 S - {x}
                backtrack(current_nums[:i] + current_nums[i+1:])
                # 回溯：撤销选择，恢复系统算符的初始状态（非常重要！），说白了是我想只用一个list就干完所有事情，而list是可变的。从本质上说跟深拷贝差不多
                path.pop()
        backtrack(nums)
        return answer
#例题：n皇后问题
```

13.动态规划问题

```python
#核心操作：递归将超时的时候用动态规划，为了节省空间，可以使用滚动数组
#例：栈混洗问题
def solve():
    n = int(input())
    if n == 0:
        print(1)
        return
    dp = [1] * (n + 2) 
    for i in range(1, n + 1):
        new_dp = [0] * (n + 2)
        for j in range(n + 1):
            if j == 0:
                new_dp[j] = dp[1]
            else:
                new_dp[j] = dp[j+1] + new_dp[j-1]
        dp = new_dp
    print(dp[0])
solve()
```

14.中序转后序

```python
def zxzhx(text):
    precedence={'+': 1, '-': 1, '*': 2, '/': 2}
    output=[]
    stack=[]
    i=0
    n=len(text)
    while i<n:
        c=text[i]
        if c.isdigit() or c=='.':
            num=[]
            while i<n and (text[i].isdigit() or text[i]=='.' ):
                num.append(text[i])
                i+=1
            output.append(''.join(num))
            continue
        elif c=='(':
            stack.append('(') 
        elif c==')':
            while stack and stack[-1]!='(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1]!='(' and precedence.get(stack[-1],0)>=precedence.get(c,0):
                output.append(stack.pop())
            stack.append(c)
        i+=1
    while stack:
        output.append(stack.pop())
    return ' '.join(output)
n=int(input())
for _ in range(n):
    text=input().strip()
    print(zxzhx(text))
```

15.归并排序

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    # 分解：将列表分成两半
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])  # 递归排序左半部分
    right_half = merge_sort(arr[mid:])  # 递归排序右半部分
    # 合并：将两个有序子列表合并
    return merge(left_half, right_half)
def merge(left, right):
    sorted_arr = []
    i = j = 0
    # 比较两个子列表的元素，按顺序合并
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_arr.append(left[i])
            i += 1
        else:
            sorted_arr.append(right[j])
            j += 1
    # 将剩余的元素添加到结果中
    sorted_arr.extend(left[i:])
    sorted_arr.extend(right[j:])
    return sorted_arr
#有时间复习一下逆序数的统计
```

16.哈希表

```python
#例：要求实现O(1)的随机存取
#绝对不能用数组，这时候应该用字典做，对于额外要求的增删应该加上双向链表
```

17.判断回文，链表就用快慢指针，列表就用双指针

18.递归+回溯

```python
#经典例题：分割回文串
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        answer = []
        path = []
        n = len(s)
        def is_palindrome(text):
            i, j = 0, len(text) - 1
            while i < j:
                if text[i] != text[j]:
                    return False
                i += 1
                j -= 1
            return True
        def backtrack(start):
            if start == n:
                answer.append(list(path))
                return
            for i in range(start + 1, n + 1):
                sub = s[start:i]
                if is_palindrome(sub):
                    path.append(sub)
                    backtrack(i)
                    path.pop()
        backtrack(0)
        return answer
    #在此基础上，如果超时，就需要动态规划回文串，进阶滚动数组
    #如果增加其他限制，不要忘了声明global 变量
```

19集合语法

```python
#集合还是满常用的，搜索为O(1)
s1 = set()			#创建空集合
s2 = set([1, 2, 3])  #从列表创建
s4 = {1, 2, 3} 		#从单个元素创建
s.add(4)   #添加元素
# 批量添加
s.update([4, 5])  # {1, 2, 3, 4, 5}
s.update((6, 7))  # 可接受任何可迭代对象
# 删除元素
s.remove(3)       # 删除3，如果不存在则报错
s.discard(10)     # 删除10，不存在也不报错
element = s.pop() # 随机删除并返回一个元素（集合无序）
s.clear()         # 清空集合
# 并集
print(A | B)      # {1, 2, 3, 4, 5, 6}
print(A.union(B)) # 相同
# 交集
print(A & B)      # {3, 4}
print(A.intersection(B))
# 差集（在A但不在B）
print(A - B)      # {1, 2}
print(A.difference(B))
# 对称差集（在A或B，但不同时在）
print(A ^ B)      # {1, 2, 5, 6}
print(A.symmetric_difference(B))
# 成员检查（O(1)）
print(1 in s)     # True
print(4 not in s) # True
# 子集/超集检查
A = {1, 2}
B = {1, 2, 3}
print(A <= B)     # True, A是B的子集
print(A.issubset(B))
print(B >= A)     # True, B是A的超集
print(B.issuperset(A))
# 1. 元素必须可哈希（hashable）
s = {1, 2, 3}        # ✓ 数字可哈希
s = {"a", "b"}       # ✓ 字符串可哈希
s = {(1, 2), (3, 4)} # ✓ 元组可哈希
s = {[1, 2], [3, 4]} # ✗ 列表不可哈希，会报错
s = {{1}, {2}}       # ✗ 集合不可哈希
```

20.类型提示

```python
def greet(name: str = None) -> str: #有类型提示，也有默认值的情形
def parse_int(s: str) -> Optional[int]:
"""可能返回整数，也可能返回None"""
#注意：像 List, Dict, Tuple, Set, Optional 这些类型需要先导入
from typing import List, Dict, Tuple, Set, Optional, Any
```

21.字典语法

```python
#创建字典
# 方法1：字面量
d = {}
d1 = {'a': 1, 'b': 2, 'c': 3}
# 方法2：构造函数
d2 = dict(a=1, b=2)  # 注意：键必须是字符串
d3 = dict([('a', 1), ('b', 2)])
# 方法3：从两个列表
keys = ['a', 'b', 'c']
values = [1, 2, 3]
d4 = dict(zip(keys, values))
#基本操作
# 添加/修改
d['apple'] = 5
d['banana'] = 3
# 读取（不存在会报KeyError）
print(d['apple'])  # 5
# 安全读取（不存在返回默认值）——非常重要，一定要加上
print(d.get('orange', 0))  # 0，自己可以改为[]或者None
# 检查键是否存在
if 'apple' in d:
    print("有苹果")
# 删除
del d['apple']
value = d.pop('banana')  # 删除并返回值
d = {'a': 1, 'b': 2, 'c': 3}
# 遍历键
for key in d:
    print(key)
# 遍历值
for value in d.values():
    print(value)
# 遍历键值对
for key, value in d.items():
    print(f"{key}: {value}")
```

22.全局变量声明

```python
GLOBAL_VAR = "地球"
def outer_lab():
    # Level 2: 国家观测站
    NONLOCAL_VAR = "中国"
    def inner_lab():
        # Level 3: 实验室内部
        LOCAL_VAR = "北京"
        # 这里能访问：
        print(LOCAL_VAR)      # ✓ 自己的仪器
        # print(NONLOCAL_VAR) # ✗ 需要声明
        # print(GLOBAL_VAR)   # ✗ 需要声明
        #尽量少用global（全局），合理使用nonlocal（外层一层）
```

23.字符串判断方法大全

| 方法         | 作用                 | 例子                     | 返回值 |
| ------------ | -------------------- | ------------------------ | ------ |
| `.isupper()` | 所有字母字符都是大写 | `"HELLO"` → `True`       | `bool` |
| `.islower()` | 所有字母字符都是小写 | `"hello"` → `True`       | `bool` |
| `.istitle()` | 每个单词首字母大写   | `"Hello World"` → `True` | `bool` |

| 方法           | 作用                                 | 例子                  | 返回值 |
| -------------- | ------------------------------------ | --------------------- | ------ |
| `.isalpha()`   | 所有字符都是字母                     | `"Hello"` → `True`    | `bool` |
| `.isdigit()`   | 所有字符都是数字                     | `"123"` → `True`      | `bool` |
| `.isalnum()`   | 所有字符都是字母或数字               | `"Hello123"` → `True` | `bool` |
| `.isdecimal()` | 所有字符都是十进制数字               | `"123"` → `True`      | `bool` |
| `.isnumeric()` | 所有字符都是数字字符（包括中文数字） | `"一二三"` → `True`   | `bool` |

| 方法         | 作用             | 例子                    | 返回值 |
| ------------ | ---------------- | ----------------------- | ------ |
| `.isspace()` | 所有字符都是空白 | `" \t\n"` → `True`      | `bool` |
| `.strip()`   | 去除两端空白     | `" hello "` → `"hello"` | `str`  |
| `.lstrip()`  | 去除左边空白     | `" hello"` → `"hello"`  | `str`  |
| `.rstrip()`  | 去除右边空白     | `"hello "` → `"hello"`  | `str`  |

| 方法              | 作用               | 例子                  | 返回值 |
| ----------------- | ------------------ | --------------------- | ------ |
| `.isascii()`      | 所有字符都是ASCII  | `"Hello"` → `True`    | `bool` |
| `.isprintable()`  | 所有字符都可打印   | `"Hello"` → `True`    | `bool` |
| `.isidentifier()` | 字符串是有效标识符 | `"var_name"` → `True` | `bool` |

24.完全二叉树问题直接用数学公式计算，不要遍历，否则会MLE

25.搜索树抓住左小右大的特点，得出中序遍历是从小到大的结论，很有用

26.查找元素

| 方法             | 适用对象 | 找不到时会怎样？    | 优点                                   | 缺点                                |
| :--------------- | :------- | :------------------ | :------------------------------------- | :---------------------------------- |
| **list.index()** | 列表     | **抛出 ValueError** | 精准，语义明确                         | 需要 `try-except` 或 `if-in` 配合   |
| **str.index()**  | 字符串   | **抛出 ValueError** | 精准，语义明确                         | 同上                                |
| **str.find()**   | 字符串   | **返回 -1**         | **安全**，无需异常处理，适合 `if` 判断 | `-1` 作为一个“魔术数字”，可读性稍差 |

27.tire树

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 字典，key是字符'0'-'9'，value是TrieNode
        self.is_end = False 
#处理字典相关问题，比如电话号码前缀
#增加孩子节点
if char not in node.children:
    node.children[char] = TrieNode()
```

28.defaultdict语法

```python
from collections import defaultdict
graph = defaultdict(list)  #当你访问一个不存在的键时，它会自动创建一个空列表 [] 作为该键的值
graph = defaultdict(int)   #自动创建整数0
graph = defaultdict(set)   #自动创建空集合()
edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
for u, v in edges:
    # 直接添加，无需检查！
    graph[u].append(v)
    graph[v].append(u)
```

29.邻接表

```python
from collections import defaultdict
#无权有向图
graph = defaultdict(list)
graph["A"].append("B")
graph["A"].append("C")
graph["B"].append("C")
# 带权无向图
graph_weighted = defaultdict(list)
edges = [('A', 'B', 5), ('A', 'C', 2), ('B', 'C', 1)]
for u, v, weight in edges:
    graph_weighted[u].append((v, weight))
    graph_weighted[v].append((u, weight))
```

30.并查集类

```python
#对于节点是否相连的问题，可以先构建出邻接表，然后DFS/BFS，只不过比较慢
#对于只要求判断联通，不要求具体路径的问题，可以使用并查集：维护一个数组，对每个节点，存储他的上司，用上司来代表一个小组都可以reach，也可以用于判断无向图是否有环路
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n  # rank[i] 表示以 i 为根的集合的秩（高度）

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # 按秩合并
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
            return True
        return False
    #！！！注意：find是方法，需要对实例调用的时候要写在实例后面！！！
```

31.Vertex类和Graph类

```python
class Vertex: #顶点类
    def __init__(self, key):
        self.id = key
        self.neighbors = {}

    def add_neighbor(self, nbr_vertex, weight=0):
        self.neighbors[nbr_vertex] = weight

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_id(self):
        return self.id

    def get_degree(self):
        return len(self.neighbors)

class Graph: #图类
    def __init__(self):
        """
        初始化一个图。
        vertices: 一个字典，key是顶点ID，value是对应的Vertex对象。
        """
        self.vertices = {}

    def add_vertex(self, key):
        """向图中添加一个新的顶点。"""
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex

    def get_vertex(self, key):
        """通过ID获取一个Vertex对象。"""
        return self.vertices.get(key)

    def add_edge(self, from_key, to_key):
        """添加一条无向边。"""
        # 确保两个顶点都存在于图中
        if from_key not in self.vertices:
            self.add_vertex(from_key)
        if to_key not in self.vertices:
            self.add_vertex(to_key)
        
        # 获取Vertex对象
        from_vertex = self.get_vertex(from_key)
        to_vertex = self.get_vertex(to_key)
        
        # 添加双向连接
        from_vertex.add_neighbor(to_vertex)
        to_vertex.add_neighbor(from_vertex)
        #单向直接删掉一个就好

    def __iter__(self):
        """让Graph对象可以被迭代，方便遍历所有Vertex对象。"""
        return iter(self.vertices.values())
```

32.遍历字母

```python
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
```

33.Dijkstra算法

```python
#有权无向图的最短路径
#Dijkstra 算法的范式：一个起点，一个距离账本，一张父节点图，一个优先队列，通过不断“松弛”邻居来更新最短路径
#例题：
import sys
import heapq # Python 的优先队列模块
from collections import defaultdict
def solve():
    # --- Part 1: 读取地点 ---
    P = int(sys.stdin.readline().strip())
    places = {sys.stdin.readline().strip() for _ in range(P)}
    # --- Part 2: 读取道路, 构建图 ---
    # 使用 defaultdict(list) 可以方便地添加边
    # 图的结构: {'地点A': [('邻居1', 距离1), ('邻居2', 距离2)], ...}
    graph = defaultdict(list)
    Q = int(sys.stdin.readline().strip())
    for _ in range(Q):
        p1, p2, dist = sys.stdin.readline().strip().split()
        dist = int(dist)
        # 无向图，所以要双向添加
        graph[p1].append((p2, dist))
        graph[p2].append((p1, dist))
    # --- Part 3: 读取并处理查询 ---
    R = int(sys.stdin.readline().strip())
    for _ in range(R):
        start_node, end_node = sys.stdin.readline().strip().split(）
        # 边界情况：起点和终点相同
        if start_node == end_node:
            print(start_node)
            continue
        # --- Dijkstra 算法实现 ---
        distances = {node: float('inf') for node in places}
        parent = {node: None for node in places}
        distances[start_node] = 0
        # 优先队列，存放 (距离, 节点)
        pq = [(0, start_node)]
        path_found = False
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            # 优化: 如果取出的节点的当前距离比账本上的大，说明是旧的、无效的记录，跳过
            if current_dist > distances[current_node]:
                continue
            # 如果取出的就是终点，说明最短路径已找到，可以提前结束
            if current_node == end_node:
                path_found = True
                break
            # 松弛操作：考察所有邻居
            for neighbor, weight in graph[current_node]:
                distance_through_current = current_dist + weight
                # 如果通过当前节点到达邻居的距离更短
                if distance_through_current < distances[neighbor]:
                    distances[neighbor] = distance_through_current
                    parent[neighbor] = current_node
                    heapq.heappush(pq, (distance_through_current, neighbor))
        # --- 回溯并格式化输出 ---
        if not path_found or distances[end_node] == float('inf'):
            # 这种情况在这题里可能不会发生，但写上更健壮
            # print("NO PATH") # 题目没要求
            pass 
        else:
            path = []
            curr = end_node
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            # 因为是倒着推的，所以要反转回来
            path.reverse()
            # 格式化输出
            output_parts = []
            for i in range(len(path) - 1):
                p_from = path[i]
                p_to = path[i+1]
                # 在图中找到这两个点之间的距离
                dist_between = -1
                for neighbor, weight in graph[p_from]:
                    if neighbor == p_to:
                        dist_between = weight
                        break
                output_parts.append(f"{p_from}->({dist_between})->")
            # 加上最后一个节点
            output_parts.append(path[-1])
            print("".join(output_parts))
solve()
```

34.字典推导式

```python
{key_expression: value_expression for item in iterable}
distances = {node: float('inf') for node in places}
```

35.无穷大

```python
float('inf')  #正无穷大
float('-inf') #负无穷大
float('nan')  #Not a Number (不是一个数)
```

36.欧拉筛

```python
def Euler_Sieve():
    MAX_SUM = 10000
    is_prime = [True] * (MAX_SUM + 1)
    primes = []
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, MAX_SUM + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > MAX_SUM:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes
primes = Euler_Sieve()
```

37.判断入度和出度

```python
#很多题要求判断图的头和尾，就用入度和出度
#例：拓扑排序判断有向图是否环路————寻找入度为0 的顶点（Kahn算法）
in_degrees = {i: 0 for i in range(n)}
# 计算所有顶点的入度
for vertex in graph:
    for neighbor in vertex.get_neighbors():
        in_degrees[neighbor.get_id()] += 1
# 将所有入度为0的顶点入队
q = deque([i for i in range(n) if in_degrees[i] == 0])
count = 0  # 记录已完成的顶点数
#然后bfs，如果全部耗尽之后count达到n，则无环
q = deque([i for i in range(n) if in_degree[i] == 0])
topo_order = []

while q:
    u = q.popleft()
    topo_order.append(u)
    for v, w in graph[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
            q.append(v)
```

38.最小生成树

```python
#寻求最短路径将所有顶点连接起来
#两种解法，prim比较常用，跟Dijkstra差不多，就是遍历点
import heapq

def prim_mst(num_nodes, edges_dict):
    """
    使用Prim算法计算最小生成树。
    :param num_nodes: 节点数量 (V)
    :param edges_dict: 邻接表表示的图，格式为 {u: [(weight, v), ...], ...}
    :return: (最小生成树的总权重, 构成MST的边列表)
    """
    if not edges_dict:
        return 0, []

    visited = [False] * num_nodes
    # 优先队列，存储 (权重, 当前节点, 源节点)
    # 源节点用于记录这条边是从哪里来的
    min_heap = [(0, 0, -1)]  # 从节点0开始，初始权重0，源节点-1表示起点
    
    mst_weight = 0
    mst_edges = []
    
    while min_heap and len(mst_edges) < num_nodes:
        weight, u, src = heapq.heappop(min_heap)
        
        # 如果节点已访问，跳过
        if visited[u]:
            continue
            
        # 标记为已访问，并加入MST
        visited[u] = True
        mst_weight += weight
        if src != -1: # 起点没有源边
            mst_edges.append((weight, src, u))
        
        # 遍历新加入节点的邻居
        for neighbor_weight, v in edges_dict.get(u, []):
            if not visited[v]:
                # 将邻居加入堆，等待被考察
                heapq.heappush(min_heap, (neighbor_weight, v, u))

    if len(mst_edges) < num_nodes - 1 and num_nodes > 1:
        return None, [] # 图不连通

    return mst_weight, mst_edges

# --- 示例 ---
if __name__ == "__main__":
    print("\n\n--- Prim 算法示例 ---")
    num_nodes_prim = 4
    # 邻接表: {u: [(权重, v), ...]}
    edges_list = [
        (10, 0, 1), (6, 0, 2), (5, 0, 3),
        (15, 1, 3), (4, 2, 3)
    ]
    adj_list = {i: [] for i in range(num_nodes_prim)}
    for w, u, v in edges_list:
        adj_list[u].append((w, v))
        adj_list[v].append((w, u)) # 无向图，两边都要加

    total_weight_prim, selected_edges_prim = prim_mst(num_nodes_prim, adj_list)

    if total_weight_prim is not None:
        print(f"最小生成树的总权重: {total_weight_prim}")
        print("构成的边:")
        for w, u, v in selected_edges_prim:
            print(f"  边({u}, {v})，权重: {w}")
            
            
#第二种：Kruskal 算法：用的并查集
class DSU:
    """
    一个带有路径压缩和按大小合并优化的并查集实现。
    """
	#略
# --------------------------------------------------
# Part 2: The Kruskal Algorithm Main Logic
# --------------------------------------------------
def kruskal_mst(num_nodes, edges):
    """
    使用Kruskal算法计算最小生成树。
    :param num_nodes: 节点数量 (V)
    :param edges: 边的列表，格式为 [(weight, u, v), ...]
    :return: (最小生成树的总权重, 构成MST的边列表)
             如果图不连通，返回 (None, [])
    """
    # 1. 将所有边按照权重从小到大排序
    edges.sort()
    
    # 2. 初始化并查集，每个节点自成一个集合
    dsu = DSU(num_nodes)
    # 3. 初始化结果
    mst_weight = 0
    mst_edges = []
    edges_count = 0
    
    # 4. 遍历排序后的边
    for weight, u, v in edges:
        # 5. 使用并查集判断加入这条边是否会形成环
        # dsu.union() 返回True意味着u和v原本不连通，可以合并
        if dsu.union(u, v):
            # 如果不形成环，就选择这条边
            mst_weight += weight
            mst_edges.append((weight, u, v))
            edges_count += 1
            # 优化：当已经选择了 V-1 条边时，MST已经构成，可以提前结束
            if edges_count == num_nodes - 1:
                break      
    # 6. 检查是否所有节点都连通了
    if edges_count < num_nodes - 1:
        print("图不连通，无法生成最小生成树。")
        return None, []
        
    return mst_weight, mst_edges
```

39.哈夫曼树

```python
#目标: 最小化带权路径长度之和
# 2. 构建哈夫曼树的函数
def build_huffman_tree(text):
    """
    输入一段文本，返回哈夫曼树的根节点。
    """
    # 统计字符频率
    frequency = Counter(text)
    
    # 创建初始的叶子节点，并放入优先队列（最小堆）
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue) # 将列表转化为最小堆
    # 如果只有一个字符，特殊处理
    if len(priority_queue) == 1:
        node = heapq.heappop(priority_queue)
        # 创建一个父节点，避免没有左右子树的情况
        merged = Node(None, node.freq)
        merged.left = node
        heapq.heappush(priority_queue, merged)
    # 当堆中元素多于1个时，循环合并
    while len(priority_queue) > 1:
        # 弹出频率最小的两个节点
        left_node = heapq.heappop(priority_queue)
        right_node = heapq.heappop(priority_queue)
        # 创建一个新的内部节点（父节点），频率是子节点之和
        # 内部节点的 char 可以是 None 或特殊符号
        merged_freq = left_node.freq + right_node.freq
        merged_node = Node(None, merged_freq)
        merged_node.left = left_node
        merged_node.right = right_node
        # 将新合并的节点放回堆中
        heapq.heappush(priority_queue, merged_node)
    # 最终，堆中只剩下一个节点，即哈夫曼树的根节点
    return priority_queue[0]
# 3. 生成哈夫曼编码的函数 (使用递归)
def generate_codes(node, prefix="", code_map={}):
    """
    遍历哈夫曼树，生成每个字符的编码。
    """
    # 如果节点是内部节点 (没有字符)，则递归地处理其子节点
    if node.char is not None:
        code_map[node.char] = prefix
    else:
        # 左边加 '0'
        if node.left:
            generate_codes(node.left, prefix + "0", code_map)
        # 右边加 '1'
        if node.right:
            generate_codes(node.right, prefix + "1", code_map)
    return code_map
#所以，哈夫曼编码的精髓就在于：用树的结构来巧妙地表示编码，并通过贪心算法构建这棵树，使其结构天然地反映出最优的编码方案。
```

