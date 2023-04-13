CREATE TABLE series_identification
(
    id_series TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE stored_map_position(
    id_series text NOT NULL PRIMARY KEY,
    time text NOT NULL,
    x REAL NULL,
    y REAL NOT NULL,
    anomaly NUMERIC
);

CREATE TABLE stored_variable_decomposition(
    id_series text NOT NULL PRIMARY KEY,
    time text NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    o REAL NOT NULL,
    ls REAL NOT NULL,
    lc REAL NOT NULL,
    ld REAL NOT NULL
);
