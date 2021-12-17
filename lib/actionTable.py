# coding=utf-8

moveTable = []
ms = ((-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1),
                (-3, 2), (-2, 3), (2, 3), (3, 2), (3, -2), (2, -3), (-2, -3), (-3, -2))

for m1 in range(58):
    for n1 in range(10):
        for l1 in range(9):
            src = n1*9+l1
            if m1<9: y = n1-(m1+1); x =l1
            elif m1<18: y = n1+(m1-8); x =l1
            elif m1 < 26: y = n1; x = l1-(m1-17)
            elif m1 < 34: y = n1; x = l1+m1-25
            elif m1 < 50:
                y = n1 + ms[m1-34][0]; x = l1+ms[m1-34][1]
            else:
                if (n1 < 3 or n1 > 6) and l1 > 2 and l1 < 6 and src%2==1-n1//5:
                    k = (m1 - 50) % 2 + 1
                    a1 = -1 if m1 < 52 or m1 > 55 else 1; a2 = 1 if m1 < 54 else -1
                    y = n1 + k * a1; x = l1 + k * a2
                    if (y > 2 and y < 7) or x < 3 or x > 5: y = -1
                else: y = -1
            moveTable.append(src*100+y*9+x if y>=0 and y<10 and x>=0 and x<9 else -1)

moveTable.append(0);
moveTable.append(10000);
moveTable.append(10001);
moveTable.append(10002);
moveTable.append(10003);

AllMoveLength = len(moveTable) # 5225

moveDict = dict(zip(moveTable,range(AllMoveLength)))
