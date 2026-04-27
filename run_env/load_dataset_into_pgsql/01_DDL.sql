CREATE SCHEMA IF NOT EXISTS car_source;

SET search_path TO car_source;

CREATE TABLE CarInfo (
    id SERIAL PRIMARY KEY,
    post_date INT,
    price BIGINT,
    mfdate INT,
    mileage FLOAT,
    gear_box VARCHAR(50),
    condition VARCHAR(50),
    fuel VARCHAR(50),
    body_style VARCHAR(50),
    origin VARCHAR(50),
    district VARCHAR(100),
    city VARCHAR(100),
    car_age INT,
    is_imported BOOLEAN
);