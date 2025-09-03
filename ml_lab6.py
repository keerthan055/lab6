

import os, math, warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import networkx as nx
import seaborn as sns

import nltk
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except:
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Try common file locations (adjust if your file is elsewhere)
candidates = [
    r"C:/Users/aksha/Downloads/stock_tweets.csv",
    "stock_tweets.csv",
    "/mnt/data/stock_tweets.csv"
]
data_file = None
for p in candidates:
    if os.path.exists(p):
        data_file = p
        break

if data_file is None:
    print("No CSV found in standard locations. Generating small synthetic dataset for testing.")
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-08-01', periods=60, freq='D')
    tickers = ['AAPL','MSFT','TSLA']
    rows = []
    for t in tickers:
        price = 100 + 10 * rng.standard_normal()
        for d in dates:
            price = price * (1 + 0.002 * rng.standard_normal())
            text = rng.choice([
                "Company reports strong earnings",
                "Product recall announced",
                "CEO resigns amid scandal",
                "Analysts upgrade stock",
                "Weak guidance reported"
            ])
            rows.append({'Date': d.strftime('%Y-%m-%d'), 'Tweet': text, 'Stock Name': t})
    df = pd.DataFrame(rows)
    print("Synthetic dataset created (shape):", df.shape)
else:
    df = pd.read_csv(data_file, low_memory=False)
    print("Loaded", data_file, " — shape:", df.shape)
print("Columns available:", df.columns.tolist())

# Normalize expected column names
col_map = {}
if 'Date' in df.columns:
    col_map['Date'] = 'date'
else:
    for c in df.columns:
        if 'date' in c.lower():
            col_map[c] = 'date'
            break
if 'Tweet' in df.columns:
    col_map['Tweet'] = 'text'
else:
    for c in df.columns:
        if 'tweet' in c.lower() or 'text' in c.lower():
            col_map[c] = 'text'
            break
if 'Stock Name' in df.columns:
    col_map['Stock Name'] = 'ticker'
else:
    for c in df.columns:
        if 'stock' in c.lower() or 'symbol' in c.lower() or 'ticker' in c.lower():
            col_map[c] = 'ticker'
            break
df = df.rename(columns=col_map)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date']).dt.date
else:
    df['date'] = pd.to_datetime(pd.date_range('2024-01-01', periods=len(df))).date
if 'text' not in df.columns:
    df['text'] = df.iloc[:, 0].astype(str)
if 'ticker' not in df.columns:
    df['ticker'] = 'TICK'

# Compute sentiment
df['sentiment_score'] = df['text'].astype(str).apply(lambda t: sia.polarity_scores(t)['compound'])

# Price column or synthetic price per ticker
price_cols = [c for c in df.columns if c.lower() in ('price','close','adjclose','adj_close','close_price')]
if price_cols:
    df['price'] = pd.to_numeric(df[price_cols[0]], errors='coerce')
else:
    df = df.sort_values(['ticker','date']).reset_index(drop=True)
    df['price'] = np.nan
    for t in df['ticker'].unique():
        idx = df['ticker']==t
        n = idx.sum()
        seed = abs(hash(str(t))) % (2**32)
        rng = np.random.RandomState(seed)
        base = 100 + 10 * rng.randn()
        steps = 1 + 0.01 * rng.randn(n)
        prices = base * np.cumprod(steps)
        df.loc[idx, 'price'] = prices

display(df.head())

# Aggregate daily per ticker
agg = df.groupby(['ticker','date']).agg(
    avg_sentiment=('sentiment_score','mean'),
    tweet_count=('sentiment_score','count'),
    mean_price=('price','mean')
).reset_index()
agg = agg.sort_values(['ticker','date']).reset_index(drop=True)

# Next-day return and labels
agg['next_mean_price'] = agg.groupby('ticker')['mean_price'].shift(-1)
agg['next_return'] = (agg['next_mean_price'] - agg['mean_price']) / agg['mean_price']
thr = 0.005
def label_return(r):
    if pd.isna(r):
        return np.nan
    if r > thr:
        return 'buy'
    elif r < -thr:
        return 'sell'
    else:
        return 'hold'
agg['label'] = agg['next_return'].apply(label_return)
agg = agg.dropna(subset=['label']).reset_index(drop=True)
print("Aggregated shape (after labeling):", agg.shape)
print(agg['label'].value_counts())

# Entropy and Gini
def entropy(labels):
    vals,counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p*math.log2(p) for p in probs if p>0])
def gini_index(labels):
    vals,counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)
print("Entropy:", entropy(agg['label'].values))
print("Gini:", gini_index(agg['label'].values))

# Binning helpers
def equal_width_binning(series, bins=4):
    edges = np.linspace(series.min(), series.max(), bins+1)
    return np.digitize(series, edges[1:-1])
def frequency_binning(series, bins=4):
    return pd.qcut(series, q=bins, labels=False, duplicates='drop')
def bin_feature(series, bins=4, bin_type='equal_width'):
    if bin_type=='equal_width':
        return equal_width_binning(series, bins=bins)
    else:
        return frequency_binning(series, bins=bins)

agg['sent_bin_eq'] = bin_feature(agg['avg_sentiment'], bins=4, bin_type='equal_width')
agg['sent_bin_freq'] = bin_feature(agg['avg_sentiment'], bins=4, bin_type='frequency')

# Information gain
def information_gain(parent_labels, subsets_labels):
    H_parent = entropy(parent_labels)
    total = sum(len(s) for s in subsets_labels)
    H_weighted = sum((len(s)/total) * entropy(s) for s in subsets_labels)
    return H_parent - H_weighted

def best_feature_by_info_gain(df_data, feature_cols, target_col, bin_continuous=True, bins=4, bin_type='equal_width'):
    base_labels = df_data[target_col].values
    best_feat, best_gain = None, -1
    for feat in feature_cols:
        series = df_data[feat]
        if pd.api.types.is_numeric_dtype(series) and bin_continuous:
            binned = bin_feature(series, bins=bins, bin_type=bin_type)
            df_data['_tmp_bin_'] = binned
            subsets = [df_data[df_data['_tmp_bin_']==v][target_col].values for v in np.unique(binned)]
        else:
            subsets = [df_data[df_data[feat]==v][target_col].values for v in np.unique(series)]
        gain = information_gain(base_labels, subsets)
        if gain > best_gain:
            best_gain = gain; best_feat = feat
    if '_tmp_bin_' in df_data.columns:
        df_data.drop(columns=['_tmp_bin_'], inplace=True)
    return best_feat, best_gain

features = ['avg_sentiment','tweet_count']
best, g = best_feature_by_info_gain(agg.copy(), features, 'label', bin_continuous=True, bins=4, bin_type='equal_width')
print("Best root feature by information gain:", best, "gain:", g)

# Simple ID3
class SimpleDecisionNode:
    def __init__(self, feature=None, children=None, is_leaf=False, label=None):
        self.feature = feature; self.children = children or {}; self.is_leaf = is_leaf; self.label = label
def majority_label(labels):
    vals,counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]
def build_id3(df_data, features_list, target_col, max_depth=5, depth=0, bin_continuous=True, bins=4, bin_type='equal_width'):
    labels = df_data[target_col].values
    if len(np.unique(labels))==1:
        return SimpleDecisionNode(is_leaf=True, label=labels[0])
    if not features_list or depth>=max_depth:
        return SimpleDecisionNode(is_leaf=True, label=majority_label(labels))
    best_feat, _ = best_feature_by_info_gain(df_data, features_list, target_col, bin_continuous, bins, bin_type)
    node = SimpleDecisionNode(feature=best_feat)
    series = df_data[best_feat]
    if pd.api.types.is_numeric_dtype(series) and bin_continuous:
        binned = bin_feature(series, bins=bins, bin_type=bin_type)
        df_data['_tmp_bin_'] = binned
        keys = np.unique(binned)
        for k in keys:
            subset = df_data[df_data['_tmp_bin_']==k]
            if subset.empty:
                node.children[k] = SimpleDecisionNode(is_leaf=True, label=majority_label(labels))
            else:
                node.children[k] = build_id3(subset, [f for f in features_list if f!=best_feat], target_col, max_depth, depth+1, bin_continuous, bins, bin_type)
        df_data.drop(columns=['_tmp_bin_'], inplace=True)
    else:
        keys = np.unique(series)
        for k in keys:
            subset = df_data[df_data[best_feat]==k]
            if subset.empty:
                node.children[k] = SimpleDecisionNode(is_leaf=True, label=majority_label(labels))
            else:
                node.children[k] = build_id3(subset, [f for f in features_list if f!=best_feat], target_col, max_depth, depth+1, bin_continuous, bins, bin_type)
    return node

tree_root = build_id3(agg.copy(), features, 'label', max_depth=4, bin_continuous=True, bins=4, bin_type='equal_width')

# Visualize custom tree
def tree_to_networkx(node, G=None, parent=None, edge_label=None, node_id=[0]):
    if G is None: G = nx.DiGraph()
    nid = node_id[0]; node_id[0]+=1
    if node.is_leaf:
        G.add_node(nid, label=f"Leaf: {node.label}")
    else:
        G.add_node(nid, label=f"Feat: {node.feature}")
    if parent is not None:
        G.add_edge(parent, nid, label=str(edge_label))
    for k,child in node.children.items():
        tree_to_networkx(child, G, nid, edge_label=k, node_id=node_id)
    return G

G = tree_to_networkx(tree_root)
plt.figure(figsize=(10,6))
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
except Exception:
    pos = nx.spring_layout(G, seed=42)
labels = nx.get_node_attributes(G,'label')
nx.draw(G, pos, with_labels=True, labels=labels, node_size=1200, font_size=8)
edge_labels = nx.get_edge_attributes(G,'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title("Custom ID3 Tree")
plt.axis('off')
plt.show()

# Sklearn Decision Tree on numeric features
X = agg[['avg_sentiment','tweet_count']].copy()
y = agg['label'].copy()
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Sklearn Decision Tree accuracy (test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

plt.figure(figsize=(12,6))
# FIX: ensure feature_names and class_names are lists
plot_tree(clf, feature_names=list(X.columns), class_names=list(le.classes_), filled=True, rounded=True)
plt.title("Sklearn Decision Tree")
plt.show()

# Decision boundary (2D)
from matplotlib.colors import ListedColormap
X_plot = X.values; y_plot = y_enc
x_min, x_max = X_plot[:,0].min()-0.01, X_plot[:,0].max()+0.01
y_min, y_max = X_plot[:,1].min()-1, X_plot[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
n_classes = len(le.classes_)
colors_light = ['#FFEEEE','#EEFFEE','#EEEEFF'][:max(1,n_classes)]
colors_bold = ['#FF0000','#00AA00','#0000FF'][:max(1,n_classes)]
cmap_light = ListedColormap(colors_light)
cmap_bold = ListedColormap(colors_bold)

plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_light)
plt.scatter(X_plot[:,0], X_plot[:,1], c=y_plot, cmap=cmap_bold, edgecolor='k')
plt.xlabel('avg_sentiment'); plt.ylabel('tweet_count')
plt.title('Decision boundary (Decision Tree)')
plt.show()
