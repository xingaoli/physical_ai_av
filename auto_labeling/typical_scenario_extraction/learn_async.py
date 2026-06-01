#!/usr/bin/env python3
"""
教学示例：理解 async/await + 事件循环 + 并发控制

模拟 2_classify_key_frame_parallel.py 的结构，但不调真实 API，
用随机 sleep 代替网络请求，打印时间线让你看到执行流程。

运行：
    python auto_labeling/typical_scenario_extraction/learn_async.py
"""

import asyncio
import random
import time


# =========================================================================
# 1. 最基础的 async/await
# =========================================================================

async def hello():
    print("    [hello] 开始")
    await asyncio.sleep(1)  # 模拟 I/O（比如发 HTTP 请求）
    print("    [hello] 结束")
    return "Hello!"


def demo_basic():
    print("=" * 60)
    print("1. 基础演示：async/await")
    print("   asyncio.run() 是桥梁")
    print("=" * 60)

    async def show_coro():
        # hello() 返回的是协程对象，还没执行
        coro = hello()
        print(f"   hello() 返回: {coro}")
        print(f"   类型: {type(coro)}")
        print("   还没执行，接下来 await 它才会真正运行...\n")
        result = await coro  # 这⾥才真正执行
        print(f"   await 后得到结果: {result}\n")

    asyncio.run(show_coro())

    # 直接 await 的更简洁写法
    print("   也可以直接 await hello():")
    result = asyncio.run(hello())
    print(f"   结果: {result}\n")


# =========================================================================
# 2. 模拟 execute_task —— 一个异步任务
# =========================================================================

async def simulate_api_call(task_id: int, duration: float):
    """模拟一个异步 API 调用。"""
    print(f"    [task {task_id}] 开始 (需要 {duration:.1f}s)")
    await asyncio.sleep(duration)  # 模拟网络延迟
    print(f"    [task {task_id}] 结束")
    return f"task-{task_id}-result"


def demo_sequential_vs_concurrent():
    print("=" * 60)
    print("2. 顺序执行 vs 并发执行")
    print("=" * 60)

    tasks_data = [
        (1, 0.5),  # (id, 耗时)
        (2, 0.3),
        (3, 0.8),
    ]

    # --- 顺序执行 ---
    print("\n   --- 顺序执行 (await 一个接一个) ---")
    t0 = time.time()

    async def sequential():
        results = []
        for tid, dur in tasks_data:
            r = await simulate_api_call(tid, dur)  # 等一个完成才启动下一个
            results.append(r)
        return results

    r = asyncio.run(sequential())
    print(f"   顺序耗时: {time.time() - t0:.2f}s (加起来 = {sum(d for _, d in tasks_data)}s)")
    print(f"   结果: {r}\n")

    # --- 并发执行 ---
    print("\n   --- 并发执行 (asyncio.gather) ---")
    t0 = time.time()

    async def concurrent():
        coros = [simulate_api_call(tid, dur) for tid, dur in tasks_data]
        # 注意: 上面只是创建了 3 个协程对象
        # gather 把它们注册到事件循环，并发调度
        results = await asyncio.gather(*coros)
        return results

    r = asyncio.run(concurrent())
    print(f"   并发耗时: {time.time() - t0:.2f}s (最慢的 = 0.8s)")
    print(f"   结果: {r}\n")


# =========================================================================
# 3. 模拟 Semaphore 控制并发度
# =========================================================================

async def execute_task_with_id(task_id: int, duration: float, semaphore: asyncio.Semaphore):
    """用信号量限制并发数的任务。"""
    async with semaphore:
        print(f"    [task {task_id}] 拿到信号量，开始 (需 {duration:.1f}s)")
        await asyncio.sleep(duration)
        print(f"    [task {task_id}] 结束，释放信号量")
        return f"task-{task_id}-result"


def demo_semaphore():
    print("=" * 60)
    print("3. Semaphore 控制并发度")
    print("   总 6 个任务，最多同时 2 个")
    print("=" * 60)

    t0 = time.time()

    async def main():
        semaphore = asyncio.Semaphore(2)  # 最多 2 个并发

        # 创建 6 个任务，每个耗时随机 0.5~1.5s
        tasks = []
        for i in range(6):
            dur = random.uniform(0.5, 1.5)
            tasks.append(execute_task_with_id(i, dur, semaphore))

        print(f"   创建了 {len(tasks)} 个协程，还没执行\n")
        results = await asyncio.gather(*tasks)
        return results

    r = asyncio.run(main())
    print(f"\n   全部完成，耗时: {time.time() - t0:.2f}s")
    print(f"   结果: {r}\n")


# =========================================================================
# 4. 完整模拟脚本结构（包含重试）
# =========================================================================

MAX_CONCURRENCY = 3
MAX_RETRIES = 2


class Task:
    """模拟 Task 类。"""
    __slots__ = ("task_id", "duration", "result", "error", "retries")

    def __init__(self, task_id: int, duration: float):
        self.task_id = task_id
        self.duration = duration
        self.result = None
        self.error = None
        self.retries = 0

    @property
    def key(self) -> str:
        return f"task-{self.task_id}"


async def call_api(task: Task) -> str:
    """模拟 VLM API 调用，有一定概率失败。"""
    print(f"      [API] {task.key} 请求中... (耗时 {task.duration:.1f}s)")
    await asyncio.sleep(task.duration)

    # 20% 概率失败
    if random.random() < 0.2:
        raise RuntimeError(f"模拟网络错误")

    return f"result-{task.task_id}"


async def execute_task(task: Task) -> Task:
    """模拟 execute_task。"""
    try:
        task.result = await call_api(task)
        task.error = None
    except Exception as e:
        task.error = str(e)
        task.result = None
    return task


async def run_tasks_with_retry(tasks: list[Task], semaphore: asyncio.Semaphore):
    """模拟 run_tasks_with_retry，和真实脚本结构一样。"""

    async def _guarded(t: Task) -> Task:
        async with semaphore:
            return await execute_task(t)

    async def _run_batch(batch: list[Task], desc: str):
        if not batch:
            return []
        coros = [_guarded(t) for t in batch]
        await asyncio.gather(*coros)
        # 打印本批结果
        for t in batch:
            status = f"OK: {t.result}" if t.result else f"FAIL: {t.error}"
            print(f"      {t.key} -> {status}")

    # 第一轮
    print(f"\n   --- 第一轮 ({len(tasks)} tasks) ---")
    await _run_batch(tasks, "initial")

    # 重试轮
    for retry_round in range(1, MAX_RETRIES + 1):
        failed = [t for t in tasks if t.error is not None]
        if not failed:
            break
        print(f"\n   --- 重试第 {retry_round} 轮 ({len(failed)} failed) ---")
        for t in failed:
            t.retries += 1
        await _run_batch(failed, f"retry-{retry_round}")

    # 最终统计
    final_failed = [t for t in tasks if t.error is not None]
    if final_failed:
        print(f"\n   警告: {len(final_failed)} 个任务最终失败:")
        for t in final_failed:
            print(f"     - {t.key}: {t.error}")


def demo_full_simulation():
    print("=" * 60)
    print("4. 完整模拟（含重试）")
    print("   10 个任务，并发度 3，最多重试 2 次")
    print("   20% 概率失败")
    print("=" * 60)

    t0 = time.time()

    # 这步是同步的：创建 Task 对象
    tasks = [Task(i, random.uniform(0.3, 1.0)) for i in range(10)]

    # asyncio.run()——从同步到异步的桥梁
    async def async_main():
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        await run_tasks_with_retry(tasks, semaphore)

    asyncio.run(async_main())

    total_time = time.time() - t0
    ok = sum(1 for t in tasks if t.result)
    fail = sum(1 for t in tasks if t.error)
    print(f"\n   全部完成! 耗时: {total_time:.2f}s, {ok} ok, {fail} failed\n")


# =========================================================================
# 5. 理解"同步包异步"的模式
# =========================================================================

def demo_sync_wrapper():
    print("=" * 60)
    print("5. 同步函数调用异步代码的模式")
    print("   这和脚本里 main() → asyncio.run(_normal_mode()) 一样")
    print("=" * 60)

    # 这是同步函数（def, 不是 async def）
    # 它不能直接 await，但可以用 asyncio.run()
    def sync_function():
        print("   [sync] 我在同步世界里，准备进入异步世界...")

        async def async_part():
            print("   [async] 我在异步世界里了!")
            await asyncio.sleep(0.5)
            print("   [async] 事情办完了，回到同步世界")

        # asyncio.run 是桥梁
        asyncio.run(async_part())

        print("   [sync] 回到同步世界，继续执行")

    sync_function()
    print()


# =========================================================================
# 6. 理解 gather 和协程创建时机
# =========================================================================

def demo_gather_timing():
    print("=" * 60)
    print("6. gather 和协程创建时机")
    print("   理解 '创建协程 ≠ 执行协程'")
    print("=" * 60)

    async def demo():
        async def say(s: str, delay: float):
            await asyncio.sleep(delay)
            print(f"      {s}")
            return s

        # 第一步：创建协程——只是创建了对象，什么都不发生
        coro1 = say("我是 coro1", 0.3)
        coro2 = say("我是 coro2", 0.1)
        print("      协程已创建，但还没人执行它们\n")

        # 第二步：await 第一个——它开始执行
        print("      await coro1...")
        r1 = await coro1  # 这会等 0.3s
        print(f"      coro1 返回: {r1}\n")

        # 第三步：再 await 第二个——它才开始
        print("      await coro2...")
        r2 = await coro2  # 这只需要等 0.1s
        print(f"      coro2 返回: {r2}\n")

        print("      对比: 用 gather 同时启动会更快:\n")

        # 对比：gather 同时启动
        coro3 = say("我是 coro3 (gather)", 0.3)
        coro4 = say("我是 coro4 (gather)", 0.1)
        await asyncio.gather(coro3, coro4)
        print("      gather 版: 两个同时跑，总耗时 ≈ 0.3s\n")

    asyncio.run(demo())


# =========================================================================
# 7. threading 辅助 asyncio：处理不支持异步的库
# =========================================================================

import time as time_module


def sync_api_call(task_id: int, duration: float) -> str:
    """模拟一个不支持异步的同步阻塞库（比如 requests、pymysql）。"""
    print(f"      [sync_api] task-{task_id} 阻塞中 ({duration:.1f}s)...")
    time_module.sleep(duration)  # 注意：这是 time.sleep，不是 asyncio.sleep！
    print(f"      [sync_api] task-{task_id} 完成")
    return f"result-{task_id}"


def demo_threading_asyncio():
    print("=" * 60)
    print("7. threading 辅助 asyncio")
    print("   场景：第三方库只有同步接口（没有 AsyncOpenAI 这种好事）")
    print("   用 run_in_executor 把阻塞调用扔到线程池")
    print("=" * 60)

    t0 = time.time()

    async def main():
        loop = asyncio.get_running_loop()

        # 模拟 6 个同步阻塞调用
        tasks_data = [(i, random.uniform(0.5, 1.5)) for i in range(6)]

        # 不用线程池的后果：直接调同步函数，卡死事件循环
        print("\n  --- 错误示范：直接调同步函数 ---")
        print("   (取消注释下面 4 行可以看到事件循环被卡死的效果)")
        print("   预期：所有 task 串行执行，Semaphore 形同虚设")
        # bad_coros = []
        # for tid, dur in tasks_data[:3]:  # 只试 3 个
        #     bad_coros.append(sync_api_call(tid, dur))  # ❌ 直接把协程变 None！
        # bad_results = await asyncio.gather(*bad_coros)  # 跑到这行直接报错

        print("   实际上上面那段根本跑不了——sync_api_call 不是 async 函数")
        print("   你不能把同步函数的返回值传给 gather 当协程用")
        print("   所以 '卡死事件循环' 的正确写法是：在一个 async 函数里调同步函数")
        print()

        # 真正展示卡死的效果：在一个 async 函数里直接调 time.sleep
        print("  --- 真正卡死的方式：async 函数里直接 time.sleep ---")

        async def bad_demo():
            print("      [bad_demo] 开始")
            time_module.sleep(1)  # ❌ 不是 await asyncio.sleep！
            print("      [bad_demo] 1s 后...")
            time_module.sleep(1)
            print("      [bad_demo] 2s 后...")

        # 如果同时跑 3 个 bad_demo，效果是串行 3s 而不是并发 1s
        async def show_blocking():
            t0 = time.time()
            print(f"\n      正常 gather 3 个 bad_demo（每个内部 time.sleep 两次共 2s）:")
            print(f"      预期：串行 6s，而不是并发的 2s")
            await asyncio.gather(bad_demo(), bad_demo(), bad_demo())
            print(f"      实际耗时: {time.time() - t0:.2f}s")
            print(f"      说明：虽然用了 gather，但 time.sleep 阻塞线程，事件循环没法做切换")

        await show_blocking()
        print()

        # 正确做法：用 loop.run_in_executor 扔到线程池
        print("\n  --- 正确做法：run_in_executor + gather ---")

        coros = []
        for tid, dur in tasks_data:
            # run_in_executor(None, func, args...)
            # None = 使用默认线程池（ThreadPoolExecutor）
            # 返回一个 asyncio.Future，可以 await
            coro = loop.run_in_executor(None, sync_api_call, tid, dur)
            coros.append(coro)

        # gather 可以正常并发！虽然是同步函数，但在不同的线程里跑
        results = await asyncio.gather(*coros)
        print(f"\n      gather 结果: {results}")

        # 还可以加上 Semaphore 控制并发线程数
        print("\n  --- 加 Semaphore 控制线程并发数 ---")
        semaphore = asyncio.Semaphore(2)

        async def _guarded_call(tid: int, dur: float):
            async with semaphore:
                return await loop.run_in_executor(None, sync_api_call, tid, dur)

        guarded_coros = [_guarded_call(tid, dur) for tid, dur in tasks_data]
        results2 = await asyncio.gather(*guarded_coros)
        print(f"\n      Semaphore 版结果: {results2}")

    asyncio.run(main())
    print(f"\n  总耗时: {time.time() - t0:.2f}s")
    print(f"  如果不靠线程池，6 个任务串行耗时 ≈ {sum(random.uniform(0.5, 1.5) for _ in range(6)):.1f}s")
    print()

    # 额外：理解 run_in_executor 的底层原理
    print("  --- run_in_executor 原理 ---")
    print("""
  loop.run_in_executor(None, fn, arg1, arg2)
       │
       ├─ 1. 创建 threading.Thread 来执行 fn(arg1, arg2)
       ├─ 2. 返回一个 asyncio.Future 对象
       ├─ 3. 线程开始运行，fn 在里面阻塞地执行
       ├─ 4. fn 返回时，线程通知事件循环："我好了"
       └─ 5. await 拿到结果，就像普通协程一样

  关键：阻塞操作在「另一个线程」里跑，不阻塞事件循环所在的线程。
        事件循环依然可以调度其他协程。
  """)

def main():
    print("\n" + "!" * 60)
    print("教学示例：理解 async/await + 并发")
    print("!" * 60 + "\n")

    demo_basic()                     # 1. 基础 async/await
    demo_sequential_vs_concurrent()  # 2. 顺序 vs 并发
    demo_semaphore()                 # 3. 信号量控制并发度
    demo_full_simulation()           # 4. 完整模拟（含重试）
    demo_sync_wrapper()              # 5. 同步包异步的模式
    demo_gather_timing()             # 6. 理解协程创建时机
    demo_threading_asyncio()         # 7. threading 辅助 asyncio

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
  async def → 定义协程函数，调用时返回协程对象（还没执行）
  await     → 交出控制权，等待异步操作完成
  asyncio.run() → 创建事件循环，跑一个协程

  asyncio.gather(*coros) → 并发执行多个协程
  asyncio.Semaphore(N)   → 限制同时执行的协程数

  关键：协程只有被 await 或交给 gather 时才会真正执行。
        没有事件循环，协程就是一堆"暂停的计算"。

  threading 在 asyncio 中的角色：
    当遇到不支持异步的同步阻塞库时（如 requests、pymysql），
    用 loop.run_in_executor() 把阻塞调用扔到线程池，
    让 threading 做 asyncio 的「辅助工人」。
""")


if __name__ == "__main__":
    main()