SET search_path TO car_source;

COPY CarInfo(post_date, price, mfdate, mileage, gear_box, condition, fuel, body_style, origin, district, city, car_age, is_imported)
FROM '/tmp/dataset/CarInfo_featured.csv'
DELIMITER ','
CSV HEADER;