def get_place(pnum):
    #points_list = sorted(list(set([x.points for x in self.players])), reverse=True) #no duplicates
    sorted_unique_points_list = sorted(list(set(points_list)), reverse=True)
    ranks = {}
    cur_place = 1
    print("Working with points list", sorted_unique_points_list)
    for pts in sorted_unique_points_list:
        ranks[pts] = cur_place
        cur_place += points_list.count(pts)

    print("Final ranks", ranks)
    return ranks[points_list[pnum]]

test_cases = [
    [100, 50, 50, 50],
    [100, 50, 50, 0],
    [100, 100, 50, 0],
    [100, 100, 100, 0]
]

points_list = test_cases[0]
assert get_place(0) == 1, get_place(0)
assert get_place(1) == 2, get_place(1)
assert get_place(2) == 2, get_place(2)
assert get_place(3) == 2, get_place(3)

points_list = test_cases[1]
assert get_place(0) == 1
assert get_place(1) == 2
assert get_place(2) == 2
assert get_place(3) == 4

points_list = test_cases[2]
assert get_place(0) == 1
assert get_place(1) == 1
assert get_place(2) == 3
assert get_place(3) == 4

points_list = test_cases[3]
assert get_place(0) == 1
assert get_place(1) == 1
assert get_place(2) == 1
assert get_place(3) == 4