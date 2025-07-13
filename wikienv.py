import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

# import wikipedia



def clean_str(p): #处理网页抓取的文本确保特殊字符的正确显示
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space): 
  def contains(self, x) -> bool: #检查给定的值是否为字符串
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    
  def _get_obs(self): #返回当前观察值
    return self.obs

  def _get_info(self):#返回当前信息
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None): #初始化状态
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword): #获取包含指定关键词的文本段落列表
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page): #获取页面的摘要信息
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity, recursion_depth=0):  # 新增递归深度参数，默认0
    # 1. 添加递归终止条件：超过最大深度（如3次）则停止
    MAX_RECURSION_DEPTH = 3
    if recursion_depth >= MAX_RECURSION_DEPTH:
        self.obs = f"无法确定实体 '{entity}' 的具体指向（已尝试多次解析歧义）。"
        return  # 终止递归

    entity_ = entity.replace(" ", "+")
    search_url = f"http://172.29.239.100:8000/viewer#search?books.name=wiki&pattern={entity_}"
    #http://172.29.239.100:8000/viewer#search?books.name=wiki&pattern=apple
    old_time = time.time()
    try:
        response_text = requests.get(search_url).text
    except Exception as e:
        self.obs = f"搜索失败：{str(e)}"
        return  # 网络错误时直接返回

    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

    if result_divs:  # 未找到精确匹配，返回相似结果
        self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        self.obs = f"未找到 '{entity}'。相似结果：{self.result_titles[:5]}。"
    else:  # 找到页面内容
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):  # 实体有歧义，需要进一步明确
            # 递归调用时，深度+1，避免无限递归
            new_entity = f"[{entity}]"
            self.search_step(new_entity, recursion_depth + 1)  # 传入当前深度+1
        else:  # 无歧义，正常提取页面内容
            self.page = ""
            for p in page:
                if len(p.split(" ")) > 2:
                    self.page += clean_str(p)
                    if not p.endswith("\n"):
                        self.page += "\n"
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  ###important:这是用于处理智能体动作
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    if self.answer is not None:  # already finished
      done = True
      return self.obs, reward, done, self._get_info()
    #关键词搜索
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    #关键词查找
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    #答案提交
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    #智能体思考
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
