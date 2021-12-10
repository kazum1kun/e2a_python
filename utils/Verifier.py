from FileReader import read_activities
from collections import Counter

s1 = read_activities('../data/activities/real/activities-real2959.txt')[1:, 1]
s2 = read_activities('../data/activities/activities-synthtest10000.txt')[1:, 1]

c1 = Counter(s1)
c2 = Counter(s2)

res1 = [(i, c1[i] / len(s1) * 100.0) for i, count in c1.most_common()]
res2 = [(i, c2[i] / len(s2) * 100.0) for i, count in c2.most_common()]

print(res1)
print(res2)