WITH id AS (
    SELECT 
        player_id
    FROM players_view
    WHERE player_name IN {}),

p1 AS (
    SELECT
        player_id as p1_id
    FROM id
    ORDER BY player_id ASC
    LIMIT 1)

SELECT
    *,
    case when p1.p1_id = winner_id then 1 else 0 end as win_p1,
    case when p1.p1_id = loser_id then 1 else 0 end as win_p2
FROM matches_view,p1
INNER JOIN id win_id ON (win_id.player_id = winner_id)
INNER JOIN id loss_id ON (loss_id.player_id = loser_id);

