import numpy


def CompetitionOpenSinceYear2int(since_year_array):
    # since_year_array is numpy array
    since_year_array[since_year_array < 2000] = 1
    since_year_array[since_year_array >= 2000] -= 1998
    return since_year_array


def split_features(X):
    X = numpy.array(X)
    X_list = []
    store_index = X[..., [0]]
    X_list.append(store_index)
    
    item_index = X[..., [1]]
    X_list.append(item_index)
    # print(X)

    day_of_week = X[..., [2]]
    X_list.append(day_of_week)

    promo = X[..., [3]]
    X_list.append(promo)

    year = X[..., [4]]
    X_list.append(year)

    month = X[..., [5]]
    X_list.append(month)

    day = X[..., [6]]
    X_list.append(day)
    
    # # transferred = X[..., [7]]
    # # X_list.append(transferred)

    family = X[..., [7]]
    X_list.append(family)
    
    clas = X[..., [8]]
    X_list.append(clas)



    perishable = X[..., [9]]
    X_list.append(perishable)

    city = X[..., [10]]
    X_list.append(city)

    cluster = X[..., [11]]
    X_list.append(cluster)

    typeO = X[..., [12]]
    X_list.append(typeO)

    oil = X[..., [13]]
    X_list.append(oil)

    state = X[..., [14]]
    X_list.append(state)
    
    isHoliday = X[..., [15]]
    X_list.append(isHoliday)
    
    isQuake = X[..., [16]]
    X_list.append(isQuake)
    
    # weekofyear = X[..., [17]]
    # X_list.append(weekofyear)


    # state_holiday = X[..., [7]]
    # X_list.append(state_holiday)

    # school_holiday = X[..., [8]]
    # X_list.append(school_holiday)

    # has_competition_for_months = X[..., [9]]
    # X_list.append(has_competition_for_months)

    # has_promo2_for_weeks = X[..., [10]]
    # X_list.append(has_promo2_for_weeks)

    # latest_promo2_for_months = X[..., [11]]
    # X_list.append(latest_promo2_for_months)

    # log_distance = X[..., [12]]
    # X_list.append(log_distance)

    # StoreType = X[..., [13]]
    # X_list.append(StoreType)

    # Assortment = X[..., [14]]
    # X_list.append(Assortment)

    # PromoInterval = X[..., [15]]
    # X_list.append(PromoInterval)

    # CompetitionOpenSinceYear = CompetitionOpenSinceYear2int(X[..., [16]])
    # X_list.append(CompetitionOpenSinceYear)

    # Promo2SinceYear = X[..., [17]] - 2008
    # Promo2SinceYear[Promo2SinceYear < 0] = 0
    # X_list.append(Promo2SinceYear)

    # State = X[..., [18]]
    # X_list.append(State)

    # week_of_year = X[..., [19]] - 1
    # X_list.append(week_of_year)

    # temperature = X[..., [20, 21, 22]]
    # X_list.append(temperature)

    # humidity = X[..., [23, 24, 25]]
    # X_list.append(humidity)

    # wind = X[..., [26, 27]]
    # X_list.append(wind)

    # cloud = X[..., [28]]
    # X_list.append(cloud)

    # weather_event = X[..., [29]]
    # X_list.append(weather_event)

    # promo_first_forward_looking = X[..., [30]] - 1
    # X_list.append(promo_first_forward_looking)

    # promo_last_backward_looking = X[..., [31]] - 1
    # X_list.append(promo_last_backward_looking)

    # stateHoliday_first_forward_looking = X[..., [32]] - 1
    # X_list.append(stateHoliday_first_forward_looking)

    # stateHoliday_last_backward_looking = X[..., [33]] - 1
    # X_list.append(stateHoliday_last_backward_looking)

    # stateHoliday_count_forward_looking = X[..., [34]]
    # X_list.append(stateHoliday_count_forward_looking)

    # stateHoliday_count_backward_looking = X[..., [35]]
    # X_list.append(stateHoliday_count_backward_looking)

    # schoolHoliday_first_forward_looking = X[..., [36]] - 1
    # X_list.append(schoolHoliday_first_forward_looking)

    # schoolHoliday_last_backward_looking = X[..., [37]] - 1
    # X_list.append(schoolHoliday_last_backward_looking)

    # googletrend_DE = X[..., [38]]
    # X_list.append(googletrend_DE)

    # googletrend_state = X[..., [39]]
    # X_list.append(googletrend_state)
    # print(len(X_list))
    return X_list