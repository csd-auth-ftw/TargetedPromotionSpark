import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

'''
FoodMart dataset preprocessing

Download FoodMart dataset from https://goo.gl/WDMoJJ and export
the 'product', 'product_class' and 'sales_fact_1997' tables as csv. 
After that, use the resulting csv files as input for this script.
'''

# global constants
VALID_CLASSES_COLUMNS = ['product_subcategory', 'product_category', 'product_department', 'product_family']
PRODUCTS_OUT_HEADER = 'product_id,categories,SRP,wholesale_price,inventory_cost\n'
CATEGORIES_OUT_HEADER = 'category_id,category_name\n'

# default options
dopt_classes_in_fname = './input/product_class.csv'
dopt_products_in_fname = './input/product.csv'
dopt_sales_in_fname = './input/sales_fact_1997.csv'
dopt_products_out_fname = './output/foodmart_products.csv'
dopt_categories_out_fname = './output/foodmart_categories.csv'
dopt_classes_columns = ['product_subcategory', 'product_category', 'product_department']

def unique(items):
    return list(set(items))

def diff(a, b):
    return list(set(a) - set(b))

def keep_valid_columns(columns):
    return list(set(columns) & set(VALID_CLASSES_COLUMNS))

def get_cost_data(sales_df, wprice_prct = 0.8):
    cost_data_df = sales_df[['product_id', 'store_sales', 'store_cost', 'unit_sales']].copy()

    # unit sales & expenses
    cost_data_df['store_sales'] = cost_data_df['store_sales'] / cost_data_df['unit_sales']
    cost_data_df['store_cost'] = cost_data_df['store_cost'] / cost_data_df['unit_sales']
    cost_data_df = cost_data_df.drop(columns=['unit_sales'])

    # compute aggregated stats about the columns
    cost_data_df = cost_data_df.groupby('product_id', as_index=False).agg({
        'store_sales': ['median'],
        'store_cost': ['min', 'max', 'median']
    }).sort_values(['product_id'])

    # compute new cost fields. Some assumptions made about wholesale_price, inventory_cost
    cost_data_df['SRP'] = cost_data_df['store_sales']['median']
    cost_data_df['wholesale_price'] = cost_data_df['store_cost']['median'] * wprice_prct
    cost_data_df['inventory_cost'] = cost_data_df['store_cost']['median'] - cost_data_df['wholesale_price']

    cost_data_df = cost_data_df[['product_id', 'SRP', 'wholesale_price', 'inventory_cost']]
    cost_data_df.columns = cost_data_df.columns.droplevel(-1)
    return cost_data_df

def create_parser():
    parser = argparse.ArgumentParser(
        description='FoodMart dataset preprocessing')

    parser.add_argument('--classes_in', default=dopt_classes_in_fname)
    parser.add_argument('--products_in', default=dopt_products_in_fname)
    parser.add_argument('--sales_in', default=dopt_sales_in_fname)
    parser.add_argument('--products_out', default=dopt_products_out_fname)
    parser.add_argument('--categories_out', default=dopt_categories_out_fname)
    parser.add_argument('-c', '--classes_columns', nargs='+', default=dopt_classes_columns)
    parser.add_argument('-d', '--classes_depth', type=int)
    parser.add_argument('-z', '--zero_index', action='store_true')
    parser.add_argument('-f', '--filter', type=int)
    parser.add_argument('-x', '--skip_filtered', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')

    return parser

def write_products(args, products_extended_df, cost_data_df, categories, index_diff):
    # join the products with the related prices
    products_extended_df = products_extended_df.merge(
        cost_data_df, on='product_id').sort_values(['product_id']).round(2)

    with open(args.products_out, 'w+') as file:
        file.write(PRODUCTS_OUT_HEADER)

        for index, row in products_extended_df.iterrows():
            product_row = []
            has_filtered = False

            for col in args.classes_columns:
                if row[col] in categories:
                    product_row.append(categories.index(row[col]) + index_diff)
                else:
                    has_filtered = True
            product_row = map(lambda i: str(i), unique(product_row))

            if args.skip_filtered and has_filtered:
                product_row = []

            file.write('{},{},{},{},{}\n'.format(
                str(row.product_id), '-'.join(product_row), row.SRP, row.wholesale_price, row.inventory_cost))

def write_categories(args, categories, index_diff):
    with open(args.categories_out, 'w+') as file:
        file.write(CATEGORIES_OUT_HEADER)
        for index in range(len(categories)):
            file.write('{},{}\n'.format(
                str(index + index_diff), categories[index]))

def main():
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
    out_dirs = map(lambda f: os.path.dirname(
        f), [args.products_out, args.categories_out])
    for d in out_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # indexing start
    index_diff = 0 if args.zero_index else 1

    # load input tables as dataframes
    classes_df = pd.read_csv(args.classes_in)
    products_df = pd.read_csv(args.products_in)
    sales_df = pd.read_csv(args.sales_in)

    # create a list of unique categories
    categories = []
    for index, row in classes_df.iterrows():
        for col in args.classes_columns:
            categories.append(row[col])
    categories = unique(categories)

    # join product with classes and keep relevant columns
    ext_mask = ['product_id', 'product_class_id'] + args.classes_columns
    products_extended_df = products_df.merge(
        classes_df, on='product_class_id').sort_values(['product_id'])[ext_mask]

    if args.filter != None or args.plot:
        # join sales with products
        sales_products_df = sales_df.merge(products_extended_df, on='product_id')

        # group and count subcategory frequencies
        subcategory_count_df = sales_products_df[['product_subcategory', 'product_id']].groupby(
            'product_subcategory', as_index=False).count().sort_values(['product_id'])

        # filter subcategories with high frequencies
        if args.filter != None:
            comp = subcategory_count_df['product_id'] >= args.filter
            high_freq_df = subcategory_count_df.where(comp).dropna(axis=0)
            blacklist = high_freq_df.product_subcategory.unique().tolist()

            # update categories
            categories = diff(categories, blacklist)

        # show plot
        if args.plot:
            subcategory_count_df.plot(kind='bar')
            plt.show()

    # compute cost data from sales
    cost_data_df = get_cost_data(sales_df)

    # save products x categories file
    write_products(args, products_extended_df, cost_data_df, categories, index_diff)

    # save categories file
    write_categories(args, categories, index_diff)

if __name__ == '__main__':
    main()
