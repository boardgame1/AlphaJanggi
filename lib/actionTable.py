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
            elif m1 < 52:
                y = n1 - (m1 -49); x = l1+(m1-49)
            elif m1 < 54:
                y = n1 + (m1 -51); x = l1+(m1-51)
            elif m1 < 56:
                y = n1 + (m1 -53); x = l1-(m1-53)
            else:
                y = n1 - (m1 -55); x = l1-(m1-55)
            moveTable.append(src*100+y*9+x if y>=0 and y<10 and x>=0 and x<9 else -1)

moveTable.append(0);
moveTable.append(10000);
moveTable.append(10001);
moveTable.append(10002);
moveTable.append(10003);

AllMoveLength = len(moveTable) # 5225

moveDict = dict(zip(moveTable,range(AllMoveLength)))
