WITH id AS (
    SELECT
        DISTINCT player
    from rankings_view
    where ranking_date >='2012-01-01' AND rank<=50
)

SELECT
    player_name
FROM players_view
INNER JOIN id ON players_view.player_id = id.player
ORDER BY player_name ASC;