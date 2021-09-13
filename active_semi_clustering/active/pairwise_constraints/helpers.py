def neighborhoods_have_cl(n1, n2, cl):
    for i in n1:
        for j in n2:
            if (i, j) in cl:
                return True
    return False

def get_constraints_from_neighborhoods(neighborhoods, oracle_ml, oracle_cl):
    ml = oracle_ml
    for neighborhood in neighborhoods:
        for i in neighborhood:
            for j in neighborhood:
                if i != j:
                    ml.append((i, j))

    cl = oracle_cl
    for neighborhood in neighborhoods:
        for other_neighborhood in neighborhoods:
            if neighborhood != other_neighborhood and neighborhoods_have_cl(neighborhood, other_neighborhood, cl):
                for i in neighborhood:
                    for j in other_neighborhood:
                        cl.append((i, j))

    return ml, cl
