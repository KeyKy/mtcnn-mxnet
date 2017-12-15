import os


less18right = 0
less18sum = 0

more18right = 0
more18sum = 0

print(__file__)


with open('/data/age-online-test/ret.txt') as f:
    lines = f.readlines()

print('sum images: %s' % len(lines))


for line in lines:
    name, age = line.split()
    age = float(age)

    if age > 99:
        continue

    if 'less18' in name:
        less18sum += 1
        if age <= 20:
            less18right += 1

    elif 'more18' in name:
        more18sum += 1
        if age > 20:
            more18right += 1


print('less18 acc: %s, cnt: %s, sum: %s' % (less18right / less18sum, less18right, less18sum))
print('more18 acc: %s, cnt: %s, sum: %s' % (more18right / more18sum, more18right, more18sum))

rightcnt = less18right + more18right
sumcnt = less18sum + more18sum
print('total acc: %s, cnt: %s, sum: %s' % (rightcnt/sumcnt, rightcnt, sumcnt))
