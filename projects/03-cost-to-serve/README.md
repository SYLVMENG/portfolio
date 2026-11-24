# Cost-to-serve customer profitability (team project)

Two-person team project. A distributor's gross margin hides the cost of picking, delivering, handling returns and answering service calls. This model allocates those costs to every order and shows which customers are actually profitable.

## Approach
- **Consolidation:** Power Query combines order, delivery, returns and service-call extracts that arrive in different layouts and date formats.
- **Allocation:** each shared cost pool is assigned to 12,000+ orders from 150+ customers by its driver (pallets by delivery zone, units picked, service minutes), using step-rated rate cards.
- **Dimensional model:** an order-level cost-to-serve fact table with customer, channel, region, product family and calendar dimensions.
- **Break-even:** VBA drives Goal Seek to find each customer's break-even order size.
- **Dashboard:** Power BI views by customer and channel.

## Results
Several high-margin customers become unprofitable once step-rated delivery and picking costs are counted. Each of the 150 customers gets one recommended action:

| Action | Customers |
|---|---|
| Keep | 97 |
| Renegotiate terms | 18 |
| Move to online channel | 24 |
| Let go | 11 |

**Tools:** Excel (Power Query, VBA, Goal Seek), Power BI (DAX).
