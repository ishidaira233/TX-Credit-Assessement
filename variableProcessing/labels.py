labelDict = {
    "account_check_status": {
        'no checking account': 0,
        '< 0 DM': 1,
        '0 <= ... < 200 DM': 2,
        '>= 200 DM / salary assignments for at least 1 year': 3,
    },

    "savings": {
        'unknown/ no savings account': 0,
        '... < 100 DM': 1,
        '500 <= ... < 1000 DM ': 2,
        '100 <= ... < 500 DM': 3,
        '.. >= 1000 DM ': 4
    },

    "present_emp_since": {
        'unemployed': 0,
        '... < 1 year ': 1,
        '1 <= ... < 4 years': 2,
        '4 <= ... < 7 years': 3,
        '.. >= 7 years': 4
    }
}


def transformTolabel(col, colName):
    return col.map(labelDict[colName])
