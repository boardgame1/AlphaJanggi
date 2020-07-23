# coding=utf-8

def create_action_labels():
    labels_array = []   # [col_src,row_src,col_dst,row_dst]
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for n1 in range(10):
        for l1 in range(9):
            destinations = [(n1, t) for t in range(9)] + \
                   [(t, l1) for t in range(10)] + \
                   [(n1 + a, l1 + b) for (a, b) in
                   [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2),
                    (-2, -3), (2, -3), (3, -2), (3, 2), (2, 3), (-2, 3), (-3, 2), (-3, -2)]]
            for (n2, l2) in destinations:
                if (n1, l1) != (n2, l2) and n2 in range(10) and l2 in range(9):
                    move = numbers[l1] + numbers[n1] + numbers[l2] + numbers[n2]
                    labels_array.append(move)

    labels_array.append('3052')
    labels_array.append('5230')
    labels_array.append('3250')
    labels_array.append('5032')
    labels_array.append('3957')
    labels_array.append('5739')
    labels_array.append('3759')
    labels_array.append('5937')

    labels_array.append('3041')
    labels_array.append('5041')
    labels_array.append('3241')
    labels_array.append('5241')
    labels_array.append('4130')
    labels_array.append('4150')
    labels_array.append('4132')
    labels_array.append('4152')

    labels_array.append('3948')
    labels_array.append('5948')
    labels_array.append('3748')
    labels_array.append('5748')
    labels_array.append('4839')
    labels_array.append('4859')
    labels_array.append('4837')
    labels_array.append('4857')

    moveTable = []
    hanList = []
    moveTable.append(10000); hanList.append(10001)
    moveTable.append(10001); hanList.append(10000)
    moveTable.append(10002); hanList.append(10002)
    moveTable.append(10003); hanList.append(10003)
    moveTable.append(0); hanList.append(0) # pass
    o0 = ord('0')
    for m in labels_array:
        moveTable.append(((ord(m[1]) - o0) * 9 + ord(m[0]) - o0) * 100 + (ord(m[3]) - o0) * 9 + ord(m[2]) - o0)
        hanList.append(((9-ord(m[1])+o0) * 9 + ord(m[0]) - o0) * 100 + (9-ord(m[3])+o0) * 9 + ord(m[2]) - o0)
    return moveTable, hanList

choList, hanList = create_action_labels()
AllMoveLength = len(choList) # 2455

choDict = dict(zip(choList,range(AllMoveLength)))
hanDict = dict(zip(hanList,range(AllMoveLength)))