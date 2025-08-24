import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data=pd.read_csv("e-commerce.csv")
df=data.dropna()
df['OrderDate']=pd.to_datetime(df['Order Date'],errors='coerce')
df['ShipDate']=pd.to_datetime(df['Ship Date'],errors='coerce')
print("----Welcome to SuperSales Data------")
while True:
    print("----Choose One Option from here:---------")
    print("1:Data Understanding & Cleaning \n2:Core KPIs (Business Metrics) \n3:Customer & Shipping Insights \n4:Visualization \n5:Automation \n6:Predictive Analysis \n7:Exit")
    user=int(input("Please Enter your choice:"))
    if user==1:
        print("------Welcome to Data Cleaning and its Understanding-----")
        print("---Shape   Info   Head-----")
        print("Shape of the Dataset is: ",df.shape)
        print("Info of the dataset is:")
        df.info()
        print("First Five rows are: \n",df.head())
    elif user==2:
        print("---Now we will do Core KPIs------")
        total_revenue=df['Sales'].sum()
        print("---Total Revenue----: \n",total_revenue)
        revenue_by_category=df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        print("---Revenue By Category from high to low-----: \n",revenue_by_category)
        revenue_by_subcat=df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False)
        print("----Revenue by Sub-Category from highest to lowest----:\n",revenue_by_subcat)
        revenue_by_region=df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        print("---Revenue by region : \n",revenue_by_region)
        top_10_products=df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False)
        print("---Top 10 products by Sales----: \n",top_10_products.head(10))
        top_10_customers=df.groupby(['Customer ID','Customer Name'])['Sales'].sum().sort_values(ascending=False)
        print("---Top 10 Customers by Sales----: \n",top_10_customers.head(10))
        df['YearMonth']=df['OrderDate'].dt.to_period('M')
        df['Quarter']=df['OrderDate'].dt.to_period('Q')
        monthly_revenue=df.groupby('YearMonth')['Sales'].sum().reset_index()
        print("---Monthly Revenue is:---- \n", monthly_revenue.head())
        quarterly_revenue=df.groupby('Quarter')['Sales'].sum().reset_index()
        print("---Quarter Revenue:----\n",quarterly_revenue.head())
    elif user==3:
        print("----Customer & Shipping Insights-----")
        sales_by_segments=df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
        print("----Sales By Segments-----: \n",sales_by_segments)
        df['Shipping_Days']=(df['ShipDate']-df['OrderDate']).dt.days
        avg_shipping_time=df['Shipping_Days'].mean()
        print("-----Average Shipping Days-----:\n",avg_shipping_time)
        shipping_by_mode=df.groupby('Ship Mode')['Shipping_Days'].mean().sort_values()
        print("-----Shipping Mode-------:\n",shipping_by_mode)
    elif user==4:
        print("----Let's Visulization all the graphs-------")
        while True:
            print("Select one option from here:")
            print("1:Bar chart → Revenue by Category \n2:Pie chart → Revenue by Segment \n3:Line chart → Monthly Revenue Trend \n4:Heatmap → Sales by Region vs Category \n5:Histogram → Shipping time distribution \n6:Exit")
            u_visula=int(input("Enter your choice you want to see: "))
            if u_visula==1:
                plt.figure(figsize=(8,5))
                sns.barplot(x=revenue_by_category.index,y=revenue_by_category.values,palette='viridis')
                plt.title("Revenue by Category")
                plt.ylabel("Revenue")
                plt.xlabel("Category")
                plt.show()
            elif u_visula==2:
                plt.pie(
                sales_by_segments.values, 
                labels=sales_by_segments.index, 
                autopct='%1.1f%%', 
                startangle=90)
                plt.title("Revenue Share by Segment")
                plt.show()

            elif u_visula==3:
                plt.figure(figsize=(12,6))
                plt.plot(monthly_revenue['YearMonth'].astype(str),monthly_revenue['Sales'],marker='o',color='blue')
                plt.title("Monthly Revenue By Trend")
                plt.xlabel("Month")
                plt.ylabel("Revenue")
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.show()
            elif u_visula==4:
                heatmap_data=pd.pivot_table(
                    data=df,
                    index='Region',
                    columns='Category',
                    values='Sales',
                    aggfunc='sum'
                )
                plt.figure(figsize=(8,5))
                sns.heatmap(heatmap_data,annot=True,fmt='.0f',cmap='YlGnBu')
                plt.title("Sales by Region vs Category")
                plt.show()
            elif u_visula==5:
                df['Shipping_Days'].hist(bins=10,edgecolor='black')
                plt.title("Shipping time distribution")
                plt.xlabel("Days")
                plt.ylabel("Number of Orders")
                plt.show()
            elif u_visula==6:
                print("---Thanks for Visualization---")
                break
            else:
                print("Please Choose Valid Option")
    elif user==5:
        print("----Automation-----")
        print("Export to Excel")
        heatmap_data = pd.pivot_table(
            data=df,
            index='Region',
            columns='Category',
            values='Sales',
            aggfunc='sum'
            )

        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
        # Create a Pandas Excel writer
        with pd.ExcelWriter("sales_summary.xlsx") as writer:
    # Total Revenue (just as an example, you can save more tables)
            total_revenue = pd.DataFrame({'Total Revenue': [df['Sales'].sum()]})
            total_revenue.to_excel(writer, sheet_name='Total Revenue', index=False)

    # Revenue by Category
            revenue_by_category.to_frame().to_excel(writer, sheet_name='Revenue by Category')

    # Average Shipping Time
            avg_shipping = pd.DataFrame({'Average Shipping Days': [df['Shipping_Days'].mean()]})
            avg_shipping.to_excel(writer, sheet_name='Avg Shipping Time', index=False)
        print("----NOW TO PDF-----")
        with PdfPages('sales_report.pdf') as pdf:
            plt.figure(figsize=(8,5))
            plt.bar(revenue_by_category.index, revenue_by_category.values, color='skyblue')
            plt.title("Revenue by Category")
            plt.xlabel("Category")
            plt.ylabel("Revenue")
            plt.xticks(rotation=45)
            pdf.savefig()   # Saves this figure to PDF
            plt.close()
    
    # 2️⃣ Monthly Revenue Trend Line Chart
            plt.figure(figsize=(12,6))
            plt.plot(monthly_revenue['YearMonth'].astype(str), monthly_revenue['Sales'], marker='o', color='blue')
            plt.title("Monthly Revenue Trend")
            plt.xlabel("Month")
            plt.ylabel("Revenue")
            plt.xticks(rotation=45)
            plt.grid(True)
            pdf.savefig()
            plt.close()
    
    # 3️⃣ Heatmap for Region vs Category
            plt.figure(figsize=(8,5))
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
            plt.title("Revenue by Region and Category")
            pdf.savefig()
            plt.close()
    elif user==6:
        print("----Welcome to Predictive Analysis-----")
        df['Shipping_Days']=(df['ShipDate']-df['OrderDate']).dt.days
        df_model=df[['Sales','Shipping_Days']].dropna()
        X=df_model[['Shipping_Days']]
        y=df_model['Sales']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        model=LinearRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        print("Prediction Values are: ",y_pred)
        print("MSE: ",mean_absolute_error(y_test,y_pred))
        print("R2Score: ",r2_score(y_test,y_pred))
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_test,y=y_pred)
        plt.title("Actual Vs Predictive")
        plt.xlabel("Actual Values")
        plt.ylabel("Predictive Values")
        plt.show()

    elif user==7:
        print("Thanks For using my System")
        break
    else:
        print("Enter a valid Option ")




# Step 4 — Visualization (Seaborn + Matplotlib)

# Heatmap → Sales by Region vs Category.

# Histogram → Shipping time distribution.

    



