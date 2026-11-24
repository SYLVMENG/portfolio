# Cost-to-serve customer profitability (team project)

Two-person team project. A distributor judges customers by gross margin, but picking, delivery, returns and service calls cost very different amounts per customer. This model allocates those costs to every order, finds who is really profitable, and recommends what to do with each account.

## Data
Synthetic data built to behave like real extracts: 150+ customers, 4 channels (including online), 3 regions, 12,000+ orders over 12 months, in four source files with different layouts (ISO dates, DD/MM/YYYY, "Mon DD YYYY", YYYYMMDD, mixed-case keys). Cost to serve was designed not to track gross margin, so some high-margin customers lose money and some low-margin ones don't.

## Model design
**1. Consolidation (Power Query).** One query per source fixes types, date formats and key casing, then the queries join on order and customer keys.

**2. Cost allocation.** Each cost pool uses its own driver and a stepped rate card:

| Cost pool | Driver | Rate structure |
|---|---|---|
| Delivery | pallets (units ÷ 90, rounded up) by zone | steps by pallet band |
| Picking | units per order | steps by unit band |
| Service calls | minutes | per minute |
| Returns | returned units | refund credited back |

**3. Dimensional model.** `Fact_OrderCTS` has one row per order with revenue, gross margin, each cost component and net profit. Dimensions: customer, channel, region, product family and calendar. A `Customer_Summary` table rolls it up.

**4. Break-even order size (VBA + Goal Seek).** Because rates step, profit per drop is not smooth, so plain Goal Seek can land on the wrong step. The macro starts Goal Seek from the last loss-making step point for each customer and checks the result.

**5. Action rule.**
- **Keep:** profitable at current order pattern.
- **Renegotiate:** break-even size is reachable (no more than half of weekly volume per drop).
- **Move to online:** a single weekly drop through the online channel would be profitable.
- **Let go:** none of the above.

## Results
53 of 150 customers are unprofitable once refunds and step-rated costs are included.

| Action | Customers |
|---|---|
| Keep | 97 |
| Renegotiate | 18 |
| Move to online | 24 |
| Let go | 11 |

## Checks
I rebuilt the allocation and the break-even search in Python to check the Excel model: cost totals matched, and all 150 break-even sizes agree with a brute-force search.

## Dashboard (Power BI)
Report pages by customer, channel and region with slicers, built on the same fact table, to show where profit is lost and which accounts need action.

The workbook and Power BI file were built jointly and are not published here.

**Tools:** Excel (Power Query, VBA, Goal Seek), Power BI (DAX), Python for validation.
