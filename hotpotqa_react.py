import os
# from openai import OpenAI
import requests
import wikienv, wrappers
import json
import sys
import random
import time
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 缓存文件路径   qwen2.5-14b-instruct
CACHE_FILE = 'hotpotqa_cache_llama70b.pkl'
# 正确推理轨迹保存路径
CORRECT_TRAJECTORIES_FILE = 'correct_trajectories.json'



# def llm(prompt, stop=None, model="qwen3-14b"):
#     client = OpenAI(
#         api_key="",
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )
#     completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {'role': 'system', 'content': 'You are a helpful assistant.'},
#             {'role': 'user', 'content': prompt}
#         ],
#         stop=stop,
#         stream=True,
#         extra_body={"enable_thinking": False},
#     )
#     full_content = ""
#     for chunk in completion:
#         if chunk.choices:
#             full_content += chunk.choices[0].delta.content
#     return full_content


def llm(prompt, stop=None):
    api_key = ""
    api_url = "https://api.together.xyz/v1/chat/completions"

    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stop":stop,
        "max_tokens": 800
    }
    time.sleep(15)
    response = requests.post(api_url, headers=headers, json=data, timeout=60,verify=False)
    result = response.json()
    # 检查是否有错误
    if "error" in result:
        error_type = result["error"].get("type", "")
        error_message = result["error"].get("message", "")
        
        # 处理频率限制错误
        if "rate_limit" in error_type.lower():
            print(f"⚠️ 频率限制错误: {error_message}")
            # 从错误信息中提取重置时间（如果有）
            retry_after = 60  # 默认等待60秒
            if "retry after" in error_message.lower():
                try:
                    retry_after = int(error_message.split("retry after ")[1].split(" ")[0]) + 5
                except:
                    pass
            print(f"等待 {retry_after} 秒后重试...")
            time.sleep(retry_after)
            raise ConnectionError("Rate limit exceeded, retrying...")
        
        # 处理其他类型的错误
        print(f"API错误 ({error_type}): {error_message}")
        raise KeyError("API返回错误格式")
    
    # 正常响应处理
    return result["choices"][0]["message"]["content"]



env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 6:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# 改进的缓存加载函数
def load_cache():
    try:
        if Path(CACHE_FILE).exists():
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                # 确保缓存有正确的结构
                if 'questions' not in cache:
                    cache['questions'] = {}
                if 'stats' not in cache:
                    cache['stats'] = {'total': 0, 'correct': 0, 'time_spent': 0.0}
                return cache
    except Exception as e:
        print(f"Warning: Failed to load cache - {str(e)}")
    return {'questions': {}, 'stats': {'total': 0, 'correct': 0, 'time_spent': 0.0}}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Warning: Failed to save cache - {str(e)}")

# 新增：保存正确推理轨迹到JSON文件
def save_correct_trajectory(question, trajectory, answer, idx):
    # 创建样本数据结构
    sample = {
        "question_id": idx,
        "question": question,
        "reasoning_trajectory": trajectory,
        "answer": answer,
        "timestamp": time.time()
    }
    
    # 读取或创建JSON文件
    if Path(CORRECT_TRAJECTORIES_FILE).exists():
        with open(CORRECT_TRAJECTORIES_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # 添加新样本
    data.append(sample)
    
    # 保存回文件
    with open(CORRECT_TRAJECTORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    cache = load_cache()
    question = env.reset(idx=idx)
    
    # 检查缓存
    if question in cache['questions']:
        cached = cache['questions'][question]
        if to_print:
            print(f"\nUsing cached result for question {idx}:")
            print(f"Question: {question}")
            print(f"Answer: {cached['info'].get('answer', 'N/A')}")
            print(f"Correct: {cached['info'].get('em', False)}")
        return cached['reward'], cached['info']
    
    if to_print:
        print(f"\nProcessing new question {idx}: {question}")
    
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    start_time = time.time()
    
    # 原有处理逻辑...
    for i in range(1, 6):

        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('Parsing error:', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    
    if not done:
        obs, r, done, info = step(env, "finish[]")
    
    elapsed_time = time.time() - start_time
    
    # 更新缓存
    cache['questions'][question] = {
        'answer': info.get('answer', ''),
        'em': info.get('em', False),
        'reward': r,
        'info': info,
        'time_spent': elapsed_time
    }
    
    # 更新统计信息
    cache['stats']['total'] += 1
    if info.get('em', False):
        cache['stats']['correct'] += 1
        # 新增：当答案正确时，保存推理轨迹
        trajectory = prompt.replace(webthink_prompt + question + "\n", "")
        save_correct_trajectory(question, trajectory, info.get('answer', ''), idx)
    
    cache['stats']['time_spent'] += elapsed_time
    
    save_cache(cache)
    
    if to_print:
        print(f"\nResult for question {idx}:")
        print(f"Answer: {info.get('answer', 'N/A')}")
        print(f"Correct: {info.get('em', False)}")
        print(f"Time spent: {elapsed_time:.2f}s")
    
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

# 主程序
if __name__ == "__main__":
    # 初始化缓存
    cache = load_cache()
    
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)
    rs = []
    infos = []
    
    for i in idxs[:1000]:
        if i !=6632:
            r, info = webthink(i, to_print=True)
            rs.append(info['em'])
            infos.append(info)
            
            # 获取当前统计信息
            cache = load_cache()
            current_total = cache['stats']['total']
            current_correct = cache['stats']['correct']
            
            # 安全地计算准确率
            accuracy = current_correct / current_total if current_total > 0 else 0
            avg_time = cache['stats']['time_spent'] / current_total if current_total > 0 else 0
            
            print(f"\nProgress: {len(rs)}/500")
            print(f"Cached questions: {len(cache['questions'])}")
            print(f"Current accuracy: {current_correct}/{current_total} = {accuracy:.2%}")
            print(f"Average time per question: {avg_time:.2f}s")
            print('-' * 40)
    
    # 最终统计
    print("\nFinal Statistics:")
    print(f"Total questions processed: {len(rs)}")
    print(f"Correct answers: {sum(rs)}")
    print(f"Accuracy: {sum(rs)/len(rs):.2%}" if rs else "N/A")
    print(f"Total time spent: {cache['stats']['time_spent']:.2f}s")
    print(f"Average time per question: {cache['stats']['time_spent']/len(rs):.2f}s" if rs else "N/A")