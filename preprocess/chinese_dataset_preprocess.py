import pandas as pd
import numpy as np


def main():
    '''
    main
    '''

    classesPath = "./data/classes.csv"
    productsPath = "./data/products.csv"

    filename_products = "chinese_products.csv"
    filename_categories = "chinese_categories.csv"

    productsDF = pd.read_csv(productsPath)
    classesDF = pd.read_csv(classesPath)

    categories = []
    for index, row in classesDF.iterrows():
        categories.append(row.product_subcategory)
        categories.append(row.product_category)
        categories.append(row.product_department)

    categories = list(set(categories))

    products_extendedDF = productsDF.merge(classesDF, on="product_class_id").sort_values(["product_id"])[["product_class_id","product_id","product_subcategory", "product_category","product_department"]]
    

    with open(filename_products,"w+") as file:
        file.write("product_id,categories" + '\n')

        for index, row in products_extendedDF.iterrows():
            r = []
            r.append(categories.index(row.product_subcategory) + 1)
            r.append(categories.index(row.product_category) + 1)
            r.append(categories.index(row.product_department) + 1)
            r = list(set(r))

            r = map(lambda value:str(value), r)
           
            file.write(str(row.product_id) + ',' + '-'.join(r) + '\n')



    with open(filename_categories, "w+") as file:
        file.write("category_id,category_name" + '\n')
        for index in range(len(categories)):
            file.write(str(index + 1) + ',' + categories[index]  + '\n')


if __name__ == "__main__":
    main()