# Codex + DeepSeek Chat Bridge

让 GPT 主代理通过 **Codex 原生多代理工具**调用 `deepseek-v4.1-flash` 子代理。DeepSeek 继续使用 Codex harness；本机 Python 网关把 Responses 转为中转站支持的 Chat Completions。

支持 macOS 与 Linux，Python 3.11+，无第三方 Python 依赖。桌面常驻安装：macOS 使用钥匙串与 LaunchAgent，Linux 使用 systemd 用户 socket；两者都只监听 `127.0.0.1`，空闲约 90 秒退出，请求到来时唤醒。不修改 Codex 应用或二进制。

已在 macOS 的 Codex `0.162.0-alpha.17.2` 验证真实 GPT→DeepSeek 工具调用、思考等级转发与本地自动压缩。Linux 的协议与安装计划由测试覆盖，首次真实设备安装仍需验证 systemd 和该设备的 Codex 版本。Codex 升级后也应重新验证。

## 在另一台设备安装

先安装 Python 3.11+、Codex，并在该设备完成 ChatGPT 登录。Linux 需要可用的 `systemctl --user`。Codex 可在 PATH 中，或用 `--codex /absolute/path/to/codex` 指定；macOS 会尝试桌面应用内置的 CLI。

```bash
gh repo clone wanfanggreat/codex-deepseek-bridge
cd codex-deepseek-bridge
python3 install.py --relay https://YOUR-RELAY.example/v1 --dry-run
python3 install.py --relay https://YOUR-RELAY.example/v1
```

将示例地址替换为自己的中转站 **base URL**，不要包含 `/chat/completions`。安装时隐藏输入 API Key；不要把 Key 放进命令参数。也支持通过已设置的 `DEEPSEEK_API_KEY` 环境变量读取。每台设备单独提供凭证，Git 不同步凭证。

安装位置是 `${CODEX_HOME:-$HOME/.codex}/deepseek-bridge`。安装会备份并修改该 Codex home 的 `config.toml`，保留现有主模型、思考等级、MCP、插件和其他角色，将默认子代理指向 DeepSeek。已有自定义 provider、模型目录、同名角色或安装目录时会拒绝覆盖；可以先用下方独立 CLI 入口。

安装成功后，完全退出并重启 Codex，再新建聊天。Linux 使用 Codex CLI 时启动新会话即可；这里不假定 Linux 已安装桌面应用。

发送：

> 请用一个全新上下文的 DeepSeek 原生子代理，以 high 思考等级读取我指定的代码文件，等待返回后汇总。主代理不要代做。若原生工具尚未显示，先发现子代理工具。

## 上下文与思考

- 默认思考等级：`high`；可选 `low / high / max`。网关实际发送 `thinking.type=enabled` 与 `reasoning_effort`，不是只修改菜单。
- 默认有效上下文：500000 tokens；自动压缩阈值：400000；每次输出上限：8192。中转站必须实际支持相应容量与输出限制，元数据不会扩大模型容量。
- DeepSeek 角色使用本地摘要压缩，不依赖中转站实现 Responses 压缩接口。32k/16k 测试中，压缩后继续工具任务、保留早期校验信息与同一子代理追问均已通过；500k 上限未做容量测试。
- 压缩会改变输入前缀，可能降低缓存命中。缓存由上游管理；网关不保存 KV 缓存。
- 混合目录使用 Codex v1 工具式多代理。v2 加密任务正文无法被第三方模型读取，网关会明确拒绝。

查看或切换窗口（如设置过 `CODEX_HOME`，下列路径要相应调整）：

```bash
python3 ~/.codex/deepseek-bridge/manage.py status
python3 ~/.codex/deepseek-bridge/set_context.py small
python3 ~/.codex/deepseek-bridge/set_context.py normal
```

`small` 为 32k 窗口、16k 阈值；`normal` 恢复 500k/400k。切换后重启并新建会话。

## 独立 CLI 入口

不修改个人配置、不安装后台服务；退出时关闭网关：

```bash
python3 launch.py --exec --relay https://YOUR-RELAY.example/v1 \
  --cwd /absolute/path/to/project --timeout 180 \
  --prompt '请用原生 DeepSeek 子代理只读检查指定文件，然后汇总。'
```

默认只读沙箱。需要修改工作目录时使用 `--sandbox workspace-write`。`--audit-dir /local/private/path` 可留存事件；Codex 事件可能包含任务内容，勿提交到 Git。

## 状态、恢复与更新

```bash
python3 ~/.codex/deepseek-bridge/manage.py status
python3 ~/.codex/deepseek-bridge/manage.py uninstall --dry-run
python3 ~/.codex/deepseek-bridge/manage.py uninstall
```

卸载会恢复配置、停止 socket 服务并删除本次安装创建的凭证；保留源码和配置备份。若之后修改了无关配置，卸载会尽量保留；手工修改过管理块时会停止自动卸载。重新安装前将旧安装目录移到备份位置，再从更新后的仓库安装。不要在运行中的服务上直接覆盖代码。

macOS API Key 保存在钥匙串；Linux API Key 保存在安装目录的 `relay.key`（用户所有、权限 0600），不是加密存储。Linux 用户服务默认随用户会话运行；本项目不会自动启用 linger 或修改系统级服务。

## 凭证与协议边界

GPT 请求只转发到官方 ChatGPT Codex 后端，保留该设备 ChatGPT 登录态。DeepSeek 请求重新构造认证头，只携带中转 Key，不转发 ChatGPT token、账户 ID 或 Cookie；禁止重定向，未知模型拒绝路由。

目前支持文本、函数工具及 `apply_patch` 转换；DeepSeek 不支持图片、官方服务端网页搜索、WebSocket 增量续聊或服务端压缩端点。独立 CLI 禁用个人 MCP、插件和记忆；桌面安装中这些能力由 GPT 主代理保留，DeepSeek 角色禁用。上游故障、截断和断流会明确报告。

仓库不包含模型目录快照、个人配置、API Key、认证文件或历史会话。安装时从目标设备 Codex 的模型目录生成本地目录，并为 DeepSeek 添加元数据。请勿提交安装目录、会话日志或备份。

## 本地测试

```bash
python3 -m unittest discover -s tests -v
```

测试不使用真实 API Key、不发起计费请求、不安装本机服务，覆盖协议转换、配置保留、凭证权限及 systemd socket 交接。当前 `gh` 令牌未授予 workflow 权限，因此仓库暂不包含 GitHub Actions 工作流；可在两种系统上运行同一条测试命令。
