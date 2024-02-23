SELECT
    tourney_date,
    tourney_name,
    surface,
    tourney_points,
    best_of,
    draw_size,
    COUNT(*) as total_matches
from matches_view
GROUP BY tourney_date,tourney_name,surface,tourney_points,best_of,draw_size
HAVING tourney_name = "{}"
ORDER BY tourney_date ASC;