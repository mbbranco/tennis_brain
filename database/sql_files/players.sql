DROP VIEW IF EXISTS players_view;

CREATE VIEW players_view AS
SELECT 
    id,
    player_id,
    name_first,
    name_last,
    name_first || " " || name_last as 'player_name',
    hand,
    dob as 'dob_int',
    ioc,
    height,
    SUBSTR(dob, 1, 4) || '-' || substr(dob, 5, 2) || '-' || substr(dob, 7, 2) as 'dob' 
FROM players
WHERE player_id IN (
    SELECT
        DISTINCT player
    FROM rankings_view
    WHERE rank <=100
);

SELECT
    * 
FROM players_view;