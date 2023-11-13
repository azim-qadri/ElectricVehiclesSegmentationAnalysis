# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# %%
df = pd.read_csv('EV_cars_India_2023 (1).csv')
df.head(30)

# %%
"""
Handle the data in car price column
"""

# %%
def clean_price_adjusted(price):
    price = price.lower().replace(' lakh', '').replace(' cr', '')
    if '-' in price:
        low, high = price.split('-')
        low = float(low.strip()) * 100 if 'cr' in low else float(low.strip())
        high = float(high.strip()) * 100 if 'cr' in high else float(high.strip())
        return (low + high) / 2
    else:
        return float(price) * 100 if 'cr' in price else float(price)

# Apply the function to the 'Car_price' column
df['Average_Price'] = df['Car_price'].apply(clean_price_adjusted)

# Optional: Segmenting the data based on price
df['Price_Segment'] = pd.cut(df['Average_Price'], bins=[0, 10, 20, float('inf')],
                                  labels=['Low', 'Medium', 'High'])

# View the first few rows to verify the changes
df.head(30)


# %%
"""
Handling rest of the data
"""

# %%
def extract_numeric(value):
    import re
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
    return float(numbers[0]) if numbers else None


# %%
df['Drive_range_km'] = df['Drive_range'].apply(extract_numeric)
df.head(21)


# %%
df['Power_Bhp'] = df['Power'].apply(extract_numeric)
df.head(20)


# %%
m = df.index[df['Charge_time'].str.contains('min')].tolist()
M = df.index[df['Charge_time'].str.contains('Min')].tolist()
for i in m:
    df.at[i,'Charge_time'] = str(extract_numeric(df.loc[i]['Charge_time'])/60)
for i in M:
    df.at[i,'Charge_time'] = str(extract_numeric(df.loc[i]['Charge_time'])/60)

df['Charge_time_hours'] = df['Charge_time'].apply(extract_numeric)

df.head(30)


# %%
df['Boot_space_L'] = df['Boot_space'].apply(extract_numeric)
df.head(20)


# %%
df['Top_speed_km'] = df['Top_speed'].apply(extract_numeric)


# %%
"""
Analyzing Battery Capacity and Drive Range in Each Price Segment
"""

# %%
# Analyzing Battery Capacity and Drive Range in Each Price Segment
battery_drive_analysis = df.groupby('Price_Segment')[['Batter_cap', 'Drive_range']].describe()
print(battery_drive_analysis)


# %%
"""
Analyzing Power and Charging Time in Each Price Segment
"""

# %%
# Analyzing Power and Charging Time in Each Price Segment
power_charge_analysis = df.groupby('Price_Segment')[['Power', 'Charge_time']].describe()
print(power_charge_analysis)


# %%
"""
Summary Statistics for Each Segment
"""

# %%
# Summary Statistics for Each Segment
summary_stats = df.groupby('Price_Segment').describe()
print(summary_stats)


# %%
"""
Price Range Distribution
"""

# %%
def plot_price_range_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Price_Segment', data=data)
    plt.title('Price Range Distribution of Electric Vehicles')
    plt.xlabel('Price Segment')
    plt.ylabel('Number of Vehicles')
    plt.show()


plot_price_range_distribution(df)


# %%
"""
Battery Capacity vs. Drive Range
"""

# %%
def plot_battery_capacity_vs_range(data):
    plt.figure(figsize=(20, 6))
    sns.scatterplot(x='Batter_cap', y='Drive_range_km', hue='Price_Segment', data=data)
    plt.title('Battery Capacity vs. Drive Range')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Drive Range (km)')
    plt.show()
plot_battery_capacity_vs_range(df)

# %%
"""
Average Power in Each Price Segment
"""

# %%
def plot_average_power_by_segment(data):
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Price_Segment', y='Power_Bhp', data=data)
    plt.title('Average Power in Each Price Segment')
    plt.xlabel('Price Segment')
    plt.ylabel('Average Power (Bhp)')
    plt.show()

plot_average_power_by_segment(df)

# %%
"""
Charge Time Distribution
"""

# %%
def plot_charge_time_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Price_Segment', y='Charge_time_hours', data=data)
    plt.title('Charge Time Distribution by Price Segment')
    plt.xlabel('Price Segment')
    plt.ylabel('Charge Time (hours)')
    plt.show()

plot_charge_time_distribution(df)

# %%
"""
Boot Space Comparison
"""

# %%
def plot_boot_space_comparison(data):
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Price_Segment', y='Boot_space_L', data=data)
    plt.title('Boot Space Comparison Across Price Segments')
    plt.xlabel('Price Segment')
    plt.ylabel('Boot Space (Liters)')
    plt.show()


plot_boot_space_comparison(df)

# %%
"""
Top Speed vs. Price
"""

# %%
def plot_top_speed_vs_price(data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Average_Price', y='Top_speed_km', data=data)
    plt.title('Top Speed vs. Price')
    plt.xlabel('Average Price (Lakhs)')
    plt.ylabel('Top Speed (kmph)')
    plt.show()

plot_top_speed_vs_price(df)

# %%
df.head()

# %%
"""
Determining optimal clusters
"""

# %%
x = df.drop(['Car_name','Car_price','Batter_cap','Drive_range','Power','Charge_time','transmission','Boot_space','Top_speed'],axis=1)
label = LabelEncoder()
x['Ps_encoded'] = label.fit_transform(x['Price_Segment'])
x.drop('Price_Segment',axis=1,inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

wcss = []  # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42,n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()





# %%
"""
we got 3 optimal clusters
"""

# %%
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42,n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)
x['Cluster'] = cluster_labels

cluster_summary = x.groupby('Cluster').mean()
print(cluster_summary)



# %%
def plot_clusters(data, x_feature, y_feature, cluster_col, centroids=None, x_idx=None, y_idx=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=cluster_col, palette='viridis')
    if centroids is not None:
        plt.scatter(centroids[:, x_idx], 
                    centroids[:, y_idx], 
                    s=100, c='red', marker='X')
    plt.title(f'{x_feature} vs {y_feature} by Cluster')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()


plot_clusters(x, 'Average_Price', 'Drive_range_km', 'Cluster')
plot_clusters(x, 'Power_Bhp', 'Charge_time_hours', 'Cluster')
