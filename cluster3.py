
import random
import numpy

import torch

import numpy as np

def calculate_center(data):
    return np.mean(data, axis=0)

def calculate_radius(data, center):
    return np.max(np.sqrt(np.sum((data - center) ** 2, axis=1)))



def get_label_and_purity(gb):


    len_label = numpy.unique(gb[:, 0], axis=0)


    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:

        num = gb.shape[0]
        gb_label_temp = {}
        for label in len_label.tolist():

            gb_label_temp[sum(gb[:, 0] == label)] = label


        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0

        label = gb_label_temp[max_label]

    return label, purity



def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    center = data_no_label.mean(axis=0)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius

def max_calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    center = data_no_label.mean(axis=0)
    radius = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius


def splits(args,gb_dict,select):
    len_of_ball = len(gb_dict)
    i = 0
    keys = list(gb_dict.keys())
    while True:
        key = keys[i]
        gb_dict_single = {key: gb_dict[key]}
        gb = gb_dict_single[key][0]
        if isinstance(gb, torch.Tensor):
            gb = gb.cpu().numpy()
        gb_dict_single[key][0] = gb

        distances = gb_dict_single[key][1]
        if isinstance(distances, np.ndarray):
            distances = distances.tolist()
        gb_dict_single[key][1] = distances

        label, p = get_label_and_purity(gb)
        number_of_samples_in_ball = len(gb_dict_single[key][0])
        if select==False:
            a_purity = args.purity_train
            a_min_ball = args.min_ball_train
        else:
            a_purity = args.purity_get_ball
            a_min_ball = args.min_ball_get_ball
        if p < a_purity and number_of_samples_in_ball > a_min_ball:
            gb_dict_new = splits_ball(gb_dict_single).copy()
            if len(gb_dict_new) > 1:
                gb_dict.pop(key)
                gb_dict.update(gb_dict_new)
                keys.remove(key)
                keys.extend(gb_dict_new.keys())
                len_of_ball += len(gb_dict_new) - 1
            else:
                i += 1
        else:
            i += 1
        if i >= len_of_ball:
            break
    return gb_dict


def calculate_distances(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2
    dis_top10 = np.sort(dis)[-10:]

    return 0.6 * np.sqrt(dis).sum() + 0.4 * np.sqrt(dis_top10).sum()
def calculate_distances2(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2


    return np.sqrt(dis.sum())

def splits_ball(gb_dict):

    center = []
    distances_other_class = []
    balls = []
    gb_dis_class = []
    center_other_class = []
    center_distances = []
    ball_list = {}
    distances_other = []
    distances_other_temp = []

    centers_dict = []
    gbs_dict = []
    distances_dict = []  #


    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):
        try:
            center.append(float(eval(center_split.strip())))
        except:
            center.append(float(center_split.strip()))
    center = np.array(center)
    centers_dict.append(center)
    gb = gb_dict_temp[1][0]
    distances = gb_dict_temp[1][1]



    len_label = numpy.unique(gb[:, 0], axis=0)

    for label in len_label.tolist():

        gb_dis_temp = []
        for i in range(0, len(distances)):
            if gb[i, 0] == label:
                gb_dis_temp.append(distances[i])
        if len(gb_dis_temp) > 0:
            gb_dis_class.append(gb_dis_temp)


    for i in range(0, len(gb_dis_class)):

        ran = random.randint(0, len(gb_dis_class[i]) - 1)
        center_other_temp = gb[distances.index(gb_dis_class[i][ran])]

        if center[0] != center_other_temp[0]:
            center_other_class.append(center_other_temp)

    centers_dict.extend(center_other_class)


    distances_other_class.append(distances)

    for center_other in center_other_class:
        balls = []
        distances_other = []
        for feature in gb:

            distances_other.append(calculate_distances(feature[1:], center_other[1:]))


        distances_other_temp.append(distances_other)
        distances_other_class.append(distances_other)



    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])

        classification = distances_temp.index(min(distances_temp))
        balls.append(classification)

    balls_array = np.array(balls)



    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])



    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])

    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1



    for i in range(len(centers_dict)):
        gb_dict_key = str(centers_dict[i][0])
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(centers_dict[i][j])
        gb_dict_value = [gbs_dict[i], distances_dict[i]]
        ball_list[gb_dict_key] = gb_dict_value

    return ball_list


def main(args,data,select):


    center_init = data[random.randint(0, len(data) - 1), :]
    distance_init = np.array([calculate_distances(feature[1:], center_init[1:]) for feature in data])


    gb_dict = {}
    gb_dict_key = str(center_init.tolist()[0])
    for j in range(1, len(center_init)):
        gb_dict_key += '_' + str(center_init.tolist()[j])

    gb_dict_value = [data, distance_init]
    gb_dict[gb_dict_key] = gb_dict_value



    gb_dict = splits(args,gb_dict,select)


    centers = []
    numbers = []
    radius = []
    max_radius = []
    lenss_ball=[]
    p1_ball=[]
    label_ball=[]

    index = []
    result = []
    if select==False:
        len_ball=1
        b_purity=0.1
    else:
        len_ball=args.min_ball_select_ball
        b_purity=args.purity_select_ball

    for i in gb_dict.keys():
        gb = gb_dict[i][0]
        if len(gb_dict[i][0]) >= len_ball:
            lab, p = get_label_and_purity(gb)
            lenss_ball.append(len(gb_dict[i][-1]))
            p1_ball.append(p)
            label_ball.append(lab)
            if p > b_purity:
                a = list(calculate_center_and_radius(gb_dict[i][0])[0])
                radius1 = calculate_center_and_radius(gb_dict[i][0])[1]
                max_radius1=max_calculate_center_and_radius(gb_dict[i][0])[1]
                lab, p = get_label_and_purity(gb_dict[i][0])
                a.insert(0, lab)

                centers.append(a)
                radius.append(radius1)
                max_radius.append(max_radius1)
                result.append(gb_dict[i][0])

                index1 = []
                for j in gb_dict[i][0]:
                    index1.append(j[0])
                numbers.append(len(gb_dict[i][-1]))
                index.append(index1)



    return numbers, result, centers, radius



def calculate_lower_radius_and_numbers(gb, center, label):

    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float32)
    center = center.to('cuda' if torch.cuda.is_available() else 'cpu')


    if not isinstance(gb, torch.Tensor):
        gb = torch.tensor(gb, dtype=torch.float32)
    gb = gb.to(center.device)

    distances = []
    for sample in gb:
        sample_label = sample[0].item()
        sample_vector = sample[1:]
        distance = torch.norm(sample_vector - center, p=2).item()
        distances.append((sample_label, distance))


    distances.sort(key=lambda x: x[1])

    lower_radius = 0
    lower_numbers = 0

    for i, (sample_label, distance) in enumerate(distances):
        if sample_label != label:
            break
        lower_radius = distance
        lower_numbers = i + 1


    total_samples = len(gb)
    probability_number = lower_numbers / total_samples if total_samples > 0 else 0

    return lower_radius, lower_numbers, probability_number
