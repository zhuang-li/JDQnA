# JDQnA

require tensorflow, json, numpy, jieba

## usage
can only work after the model training is done.

query

```python qna.py --query "什么是满返满赠？"```

result 

```
满 返 是 指 购物 购物满 一定 定金 金额 的 情况 下 减去 订单 部分 分款 款项   返 优惠 优惠券 或者 京 豆 的 促销 方式    满 赠 是 指 购物 购物满 一定 定金 金额 的 情况 下 获得 赠品   赠品 分为 自动 赠送 以及 购物 购物车 领取 两种 形式    注   满 返 满 赠 活动 细则 还 请 以 您 参与 的 具体 活动 规则 为准  
若   商品 页面 面有 满 减 活动   下单 后 订单 金额 没有 有变 变化    可能 存在 以下 几种 情况   1   订单 的 金额 没有 达到 满 减 活动 的 金额 要求   2   订单 中 有 未 参加 加满 减 活动 的 商品   3   所选 选购 的 商品 享受 多个 促销   如   满 减 或者 是 享受 受赠 赠品   加入 购物 购物车 时 默认 的 活动 是 享受 受赠 赠品 的 情况 下   此时 需 在 购物 购物车 车商 商品 商品价格 价格 下方 有 一个   修 促销 优惠   的 红色 字体 下拉 下拉框   点击 选择 参加 加满 减 活动 即可   同理 享受 受赠 赠品 不 参加 加满 减 也 可以 按照 照此 方法 操作    4   若以 以上 情况 都 不存 存在   请 删除 购物 购物车 中 的 商品   重新 登陆 后 购买  
  返 券 试用     是 先付 付款 购买 试用 商品   然后 在 规定 定时 时间 内 提交 原创 且 优质 的 试用 报告   经 审核 通过 过后 再 返还 与 商品 订单 实付 金额 等额   不含 运费   的 京 券 到 您 的 京东 账号 的 活动      第三 第三方 三方 入驻 的 商家 店铺 的 试用 商品   返还 的 是 店铺 京 券      返 券 试用 的 流程 如下          这里 有 很多 试用 商品    
由于 活动 优惠 方式 或 促销 种类 非常 多   具体 规则 都 不 相同   您 所购 购买 的 商品 实际 能 享受 优惠 请 以 结算 页面 为准    例如   1  如 满 减 活动 未 减价   可能 由于 您 购买 的 商品 中 有 套装 商品   或者 有 赠品 都 无法 参加 加满 减   2  赠品 未领 领取 活动   商品 赠品 有 满 赠 活动 一般 都 会 在 购物 购物车 中 显示   领取 赠品   的 红色 提示 按钮   需 点击 领取   如 赠品 无需 领取   会 在 购物 购物车 车商 商品 下方 方显 显示 赠品  xx  赠品 品名 名称    下单 后 都 是 会 直接 接在 订单 中 显示 赠品 商品   您 可以 根据 您 订单 所选 选择 的 商品 进行 确认  
您好   余额 返 卡 退款 周期 如下  

```
## Proposed Method

Now we just calculate the query similarity between the query and the QA pair. And the query which is in the same category with the question would have a higher score.

## Observation

For now, it seems the classification model is not much helpful as there are not enough training data (we only have 783 QA pair). But the simple tfidf method works way much better. I think it is because on the small-scale data, the hand-crafted features include more expert knowledge so work better.

## To do

Try to calculate the query similarity with word embeeding or a siamese network instead of the tfidf and see if it can achieve a better performance.
