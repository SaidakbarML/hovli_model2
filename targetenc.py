class TargetEncoding:
    def init(self, alpha=10):
        self.alpha = alpha
        self.global_mean = None
        self.mappings = {}

    def fit(self, df, cat_cols, target_col):
        self.global_mean = df[target_col].mean()
        for col in cat_cols:
            stats = df.groupby(col)[target_col].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            smooth = (counts * means + self.alpha * self.global_mean) / (counts + self.alpha)
            self.mappings[col] = smooth

    def transform(self, df, cat_cols):
        df = df.copy()
        for col in cat_cols:
            df[col + '_tencoded'] = df[col].map(self.mappings[col])
            df[col + '_tencoded'] = df[col + '_tencoded'].fillna(self.global_mean)

        # Drop all original categorical columns at once after encoding
        df = df.drop(columns=cat_cols)
        return df

    def fit_transform(self, df, cat_cols, target_col):
        self.fit(df, cat_cols, target_col)
        return self.transform(df, cat_cols)