SELECT
    rank,
    ranking_date,
    points
from rankings_view
where player = (
    SELECT player_id from players_view
    where player_name = "{}"
)
GROUP BY player
having ranking_date = MAX(ranking_date);