from bs4 import BeautifulSoup
from datetime import date,timedelta
import urllib.request


def get_img_link(link):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    info = opener.open(link).read()
    soup = BeautifulSoup(info, "html.parser")
    i = soup.find(class_='player-profile-hero-image')
    img_link = 'https://www.atptour.com' + i.img['src']
    return img_link

def get_current_ranking_photo(p1,p2):
    players,links = get_updated_ranking()
    p1_rank,p2_rank,p1_link,p2_link = None,None,None,None
    if p1 in players:
        p1_index = players.index(p1)
        p1_link = get_img_link(links[p1_index])
        p1_rank = p1_index+1
    if p2 in players:
        p2_index = players.index(p2)
        p2_link = get_img_link(links[p2_index])
        p2_rank = p2_index+1

    return p1_rank, p2_rank, p1_link, p2_link


def get_updated_ranking():
    rank_start = list(range(0,1601,100))
    rank_finish = [r + 100 for r in rank_start]

    rank_range = []
    for i in range(len(rank_start)):
        rank_range.append(f'{rank_start[i]}-{rank_finish[i]}')

    # today_date = date.today()
    # date_ranking = today_date - timedelta(days=today_date.weekday())

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    players_ranking = []
    links_ranking = []
    rank_range = rank_range[0:2]
    for r in rank_range:
        url = f'https://www.atptour.com/en/rankings/singles?rankRange={r}'
        #url = f'https://www.atptour.com/en/rankings/singles?rankRange={r}&rankDate={date_ranking}'
        info = opener.open(url).read()
        soup = BeautifulSoup(info, "html.parser")
        players, links = scrape_it(soup)
        players_ranking.extend(players)
        links_ranking.extend(links)

    return players_ranking, links_ranking

def scrape_it(soup):
    names = soup.find_all(class_='player-cell-wrapper')
    players = []
    links = []
    for name in names:
        p = name.find('a').contents[0].strip()
        link = name.find('a')['href']

        spl = p.split(' ')
        players.append(spl[-1]+" "+spl[0][0]+".")
        links.append('https://www.atptour.com' + link)

    return players,links


if __name__=='__main__':
    # get_updated_ranking()
    p1_rank,p2_rank,p1_link,p2_link = get_current_ranking_photo('Alcaraz C.','Nadal R.')
    print(p1_rank,p2_rank)
    print(p1_link,p2_link)