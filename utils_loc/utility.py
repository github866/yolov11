def xywh_to_xyxy(xywh):
    """
    Convert xywh to xyxy format.
    :param x: (x, y, w, h)
    :return: (x1, y1, x2, y2)
    """
    if len(xywh) != 4:
        return None
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[0] + xywh[2]
    y2 = xywh[1] + xywh[3]
    return int(x1), int(y1), int(x2), int(y2)

def id_to_name(id):
    """
    Convert id to name.
    :param id: id
    :return: name
    """
    if id == 1:
        return "Nurse_1"
    elif id == 2:
        return "Nurse_2"
    elif id == 3:
        return "Patient_1"
    elif id == 4:
        return "Patient_2"
    elif id == 5:
        return "Patient_3"
    elif id == 6:
        return "Psychiatrist"
    elif id == 7:
        return "Psychologist"
    elif id == 11:
        return "Researcher"
    elif id == 8:
        return "Person_1"
    elif id == 9:
        return "Person_2"
    elif id == 10:
        return "Person_3"
    elif id == 12:
        return "Person_4"

def color_to_room(rgb: tuple):
    if rgb == (255, 0, 0): #blue
        return "CSU Miliue"
    elif rgb == (0, 255, 0): #green
        return 'Nursing Station'
    elif rgb == (0, 0, 255): #red
        return 'Quiet Room'
    elif rgb == (255, 255, 0): #cyan
        return 'Sally Port / Entrance'
    elif rgb == (0, 255, 255): #yellow
        return 'Unknown Room'