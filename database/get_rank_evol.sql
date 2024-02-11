WITH id AS (
    SELECT 
        player_id
    FROM players_view
    WHERE player_name = "{}"
)

SELECT
    rankings_view.player,
    rankings_view.ranking_date,
    rankings_view.rank,
    rankings_view.points
from rankings_view, id
WHERE id.player_id = rankings_view.player
ORDER BY rankings_view.player, rankings_view.ranking_date;