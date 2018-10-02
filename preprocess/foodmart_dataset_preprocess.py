import pandas as pd
import numpy as np
import os
import argparse

'''
FoodMart dataset preprocessing

Download FoodMart dataset from https://goo.gl/WDMoJJ and export
the 'product' and 'product_class' tables as csv. After that, use the
resulting csv files as input for this script.
'''

# global constants
VALID_CLASSES_COLUMNS = ['product_subcategory', 'product_category', 'product_department', 'product_family']
PRODUCTS_OUT_HEADER = 'product_id,categories\n'
CATEGORIES_OUT_HEADER = 'category_id,category_name\n'

# default options
dopt_classes_in_fname = './input/product_class.csv'
dopt_products_in_fname = './input/product.csv'
dopt_products_out_fname = './output/foodmart_products.csv'
dopt_categories_out_fname = './output/foodmart_categories.csv'
dopt_classes_columns = ['product_subcategory', 'product_category', 'product_department']

def unique(items):
    return list(set(items))

def keep_valid_columns(columns):
    return list(set(columns) & set(VALID_CLASSES_COLUMNS))

def create_parser():
    parser = argparse.ArgumentParser(
        description='FoodMart dataset preprocessing')

    parser.add_argument('--classes_in', default=dopt_classes_in_fname)
    parser.add_argument('--products_in', default=dopt_products_in_fname)
    parser.add_argument('--products_out', default=dopt_products_out_fname)
    parser.add_argument('--categories_out', default=dopt_categories_out_fname)
    parser.add_argument('-c', '--classes_columns', nargs='+', default=dopt_classes_columns)
    parser.add_argument('-d', '--classes_depth', type=int)
    parser.add_argument('-z', '--zero_index', action='store_true')

    return parser

def main():
    '''
    main
    '''

    # parse cli args
    args = create_parser().parse_args()

    # set columns using depth parameter
    if args.classes_depth != None:
        args.classes_columns = VALID_CLASSES_COLUMNS[:args.classes_depth]
    args.classes_columns = keep_valid_columns(args.classes_columns)

    # check if columns are empty
    if len(args.classes_columns) < 1:
        print('ERROR: no valid columns selected')
        exit(1)

    # create missing output directories
    out_dirs = map(lambda f: os.path.dirname(f), [args.products_out, args.categories_out])
    for d in out_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # indexing start
    index_diff = 0 if args.zero_index else 1

    # load input tables as dataframes
    classes_df = pd.read_csv(args.classes_in)
    products_df = pd.read_csv(args.products_in)

    # create a list of unique categories
    categories = []
    for index, row in classes_df.iterrows():
        for col in args.classes_columns:
            categories.append(row[col])
    categories = unique(categories)

    # join product with classes and keep relevant columns
    ext_mask = ['product_id', 'product_class_id'] + args.classes_columns
    products_extended_df = products_df.merge(classes_df, on='product_class_id').sort_values(['product_id'])[ext_mask]

    # save products x categories file
    with open(args.products_out, 'w+') as file:
        file.write(PRODUCTS_OUT_HEADER)

        for index, row in products_extended_df.iterrows():
            product_row = []
            for col in args.classes_columns:
                product_row.append(categories.index(row[col]) + index_diff)
            product_row = map(lambda i: str(i), unique(product_row))

            file.write('{},{}\n'.format(str(row.product_id), '-'.join(product_row)))

    # save categories file
    with open(args.categories_out, 'w+') as file:
        file.write(CATEGORIES_OUT_HEADER)
        for index in range(len(categories)):
            file.write('{},{}\n'.format(str(index + index_diff), categories[index]))

if __name__ == '__main__':
    main()
