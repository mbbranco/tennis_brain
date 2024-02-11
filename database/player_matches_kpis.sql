WITH id AS (
    SELECT 
        player_id,
        player_name 
    FROM players_view
    WHERE player_name = "{}"
),

winner AS(
    SELECT
        *,
        1 as 'win',
        0 as 'loss'
    FROM matches_view, id
    WHERE id.player_id = winner_id
),

loser AS(
    SELECT
        *,
        0 as 'win',
        1 as 'loss'
    FROM matches_view, id
    where id.player_id = loser_id
),

all_matches AS(
    SELECT * FROM winner
    UNION
    SELECT * FROM loser
),


kpis AS (
    SELECT 
        id,
        tourney_date,
        tourney_year,
        tourney_name,
        round,
        round_level,
        tourney_points,
        surface,
        match_num,
        player_id,
        player_name,
        win,
        loss,
        SUM(win) OVER (ROWS UNBOUNDED PRECEDING) as wins_cumsum,
        SUM(loss) OVER (ROWS UNBOUNDED PRECEDING) as losses_cumsum,
        SUM(win) OVER (ORDER BY tourney_date, match_num ASC ROWS 9 PRECEDING) as wins_last10,
        SUM(loss) OVER (ORDER BY tourney_date,match_num ASC ROWS 9 PRECEDING) as losses_last10
    FROM all_matches
    GROUP BY tourney_date,match_num
    ORDER BY tourney_date,match_num
)

SELECT
    *,
    ROUND(CAST(wins_cumsum AS FLOAT)/losses_cumsum,2) as win_loss_ratio_start,
    ROUND(CAST(wins_cumsum AS FLOAT)/(wins_cumsum+losses_cumsum),2) as win_perc_start,
    ROUND(CAST(wins_last10 AS FLOAT)/losses_last10,2) as win_loss_ratio_last10,
    ROUND(CAST(wins_last10 AS FLOAT)/(wins_last10+losses_last10),2) as win_perc_last10
FROM kpis;