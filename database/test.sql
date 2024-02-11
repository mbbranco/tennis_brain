WITH id AS (
    SELECT 
        *
    FROM players_view
    WHERE player_name = "{}"
)

SELECT
    player,
    ranking_date,
    rank,
    points
from rankings_view
INNER JOIN id ON id.player_id = rankings_view.player
ORDER BY player, ranking_date;

