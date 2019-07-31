import requests
import json
import datetime
import pandas as pd
from pprint import pprint

import pos_tag


max_page = 20
countPerPage = 50

directory = {
    "교육, 학문": 11,
    "컴퓨터통신": 1,
    "게임": 2,
    "엔터테인먼트, 예술": 3,
    "생활": 8,
    "건강": 7,
    "사회, 정치": 6,
    "경제": 4,
    "여행": 9,
    "스포츠, 레저": 10
}

selected_dir = "게임"

save_data = []


def process(sel_dir):
    with requests.Session() as sess:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.48 Safari/537.36 Edg/74.1.96.24"
        res = sess.get("https://kin.naver.com/qna/questionList.nhn", headers={"user-agent": user_agent})
        print(res.cookies)
        print(res.headers)
        cookie = res.headers["Set-Cookie"]
        headers = {
            "authority": "kin.naver.com",
            "path": "/ajax/mainNoanswer.nhn?m=noanswer&page=1&dirId=1&selTab=qna&queryTime=2019-04-11%2017%3A17%3A08&countPerPage=50&viewType=onlyTitle&sort=answer",
            "cookie": cookie,
            "user-agent": user_agent,
            "referer": "https://kin.naver.com/qna/questionList.nhn"
        }

        for page in range(1, max_page+1):
            now_time = datetime.datetime.now()
            print(now_time.strftime("%Y-%m-%d %H:%M:%S"))

            res = sess.get("https://kin.naver.com/ajax/mainNoanswer.nhn?m=noanswer&page=" + str(page) + "&dirId=" + str(directory[sel_dir]) + "&selTab=qna&queryTime=" + now_time.strftime("%Y-%m-%d %H:%M:%S") + "&countPerPage=" + str(countPerPage) + "&viewType=onlyTitle", headers=headers)

            print("================================")
            data = json.loads(res.content.decode())
            print("카테고리:", data["result"][0]["selDirName"])
            for key, val in data["result"][0]["noanswer"].items():
                if key == "countPerPage":
                    print("countPerPage:", val)
                elif key == "list":
                    for i in val:
                        # save_data.append(i["title"])
                        if i["previewContents"] not in save_data:
                            save_data.append(i["previewContents"])
                            print("title:", i["title"])
                            print("previewContents", i["previewContents"])
                            print("|----------|")
                    print("num:", len(val))


if __name__ == "__main__":
    while True:
        print("[ 다음 카테고리중 선택해주세요. ]")
        pprint(list(directory.keys()))
        selected_dir = input("입력:")
        if selected_dir == 'Q':
            print('=== 종료 ===')
            series = pd.DataFrame(save_data, columns=["question"])
            series.drop_duplicates(['question'], keep='last')
            # series.to_csv("data.csv", header=True, index=False)

            with open("train_data.txt", "w", encoding="utf-8") as f:
                f.write(" ".join([i for x in pos_tag.new_pos(series.values, discard_stopwords=True) for i in x]))
            break
        if selected_dir not in directory:
            continue

        process(selected_dir)