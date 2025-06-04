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
        return "Nurse1"
    elif id == 2:
        return "Nurse2"
    elif id == 3:
        return "Patient1"
    elif id == 4:
        return "Patient2"
    elif id == 5:
        return "Patient3"
    elif id == 6:
        return "Psychiatrist"
    elif id == 7:
        return "Psychologist"
    else:
        return f"Unknown {id}"

def color_to_room(rgb: tuple):
    if rgb == (255, 0, 0): #blue
        return "CSU_Miliue"
    elif rgb == (0, 255, 0): #green
        return 'Nursing Station'
    elif rgb == (0, 0, 255): #red
        return 'Quiet Room'
    elif rgb == (255, 255, 0): #cyan
        return 'Sally_Port_Entrance'
    elif rgb == (0, 255, 255): #yellow
        return 'Unknown Room'