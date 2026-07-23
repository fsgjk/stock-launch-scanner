"""
股票池管理模块
"""
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger

from utils.database import get_db


class StockPoolManager:
    """股票池管理器"""

    # 预定义分组
    DEFAULT_GROUPS = ["默认分组", "重点关注", "短线操作", "中长线", "观察列表"]

    def add_stock(self, code: str, name: str = "", group: str = "默认分组",
                  notes: str = "") -> bool:
        """添加股票到股票池"""
        code = str(code).zfill(6)
        try:
            with get_db() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO stock_pool (code, name, group_name, notes)
                    VALUES (?, ?, ?, ?)
                """, (code, name, group, notes))
            logger.info(f"添加股票 {code} {name} 到 {group}")
            return True
        except Exception as e:
            logger.error(f"添加股票失败: {e}")
            return False

    def remove_stock(self, code: str, group: str = None) -> bool:
        """从股票池移除股票"""
        code = str(code).zfill(6)
        try:
            with get_db() as conn:
                if group:
                    conn.execute(
                        "DELETE FROM stock_pool WHERE code=? AND group_name=?",
                        (code, group)
                    )
                else:
                    conn.execute("DELETE FROM stock_pool WHERE code=?", (code,))
            logger.info(f"移除股票 {code}")
            return True
        except Exception as e:
            logger.error(f"移除股票失败: {e}")
            return False

    def update_stock(self, code: str, group: str = None, notes: str = None,
                     is_active: int = None) -> bool:
        """更新股票信息"""
        code = str(code).zfill(6)
        updates = []
        params = []
        if group is not None:
            updates.append("group_name=?")
            params.append(group)
        if notes is not None:
            updates.append("notes=?")
            params.append(notes)
        if is_active is not None:
            updates.append("is_active=?")
            params.append(is_active)
        if not updates:
            return False
        params.append(code)

        try:
            with get_db() as conn:
                conn.execute(
                    f"UPDATE stock_pool SET {', '.join(updates)} WHERE code=?",
                    params
                )
            return True
        except Exception as e:
            logger.error(f"更新股票失败: {e}")
            return False

    def get_pool(self, group: str = None, active_only: bool = True) -> pd.DataFrame:
        """获取股票池列表"""
        with get_db() as conn:
            if group:
                rows = conn.execute(
                    "SELECT * FROM stock_pool WHERE group_name=? AND is_active=1",
                    (group,)
                ).fetchall()
            elif active_only:
                rows = conn.execute(
                    "SELECT * FROM stock_pool WHERE is_active=1 ORDER BY group_name, code"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM stock_pool ORDER BY group_name, code"
                ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["id", "code", "name", "market", "group_name",
                                         "added_date", "is_active", "notes"])
        return pd.DataFrame([dict(r) for r in rows])

    def get_codes(self, group: str = None) -> List[str]:
        """获取股票池中的代码列表"""
        df = self.get_pool(group=group)
        if df.empty:
            return []
        return df["code"].tolist()

    def get_groups(self) -> List[str]:
        """获取所有分组"""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT group_name FROM stock_pool WHERE is_active=1"
            ).fetchall()
        groups = [r["group_name"] for r in rows]
        return groups if groups else self.DEFAULT_GROUPS

    def create_group(self, group_name: str) -> bool:
        """创建新分组"""
        # 分组通过添加股票时自动创建，这里只做校验
        if not group_name or not group_name.strip():
            return False
        return True

    def import_from_list(self, stocks: List[Dict], group: str = "默认分组"):
        """批量导入股票"""
        count = 0
        for s in stocks:
            if self.add_stock(
                code=s.get("code", s.get("代码", "")),
                name=s.get("name", s.get("名称", "")),
                group=group,
                notes=s.get("notes", "")
            ):
                count += 1
        logger.info(f"批量导入完成: {count}/{len(stocks)}")
        return count

    def get_pool_summary(self) -> Dict:
        """获取股票池概览"""
        df = self.get_pool()
        if df.empty:
            return {"total": 0, "groups": {}, "codes": []}

        summary = {
            "total": len(df),
            "groups": df["group_name"].value_counts().to_dict(),
            "codes": df["code"].tolist(),
            "names": dict(zip(df["code"], df["name"])),
        }
        return summary

    def to_csv(self, filepath: str = None):
        """导出股票池到CSV"""
        df = self.get_pool(active_only=False)
        if filepath is None:
            filepath = f"stock_pool_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"股票池导出到 {filepath}")
        return filepath


# 单例
pool_manager = StockPoolManager()


if __name__ == "__main__":
    from utils.database import init_database
    init_database()

    # 测试：添加一些示例股票
    samples = [
        {"code": "600519", "name": "贵州茅台", "group": "重点关注"},
        {"code": "000858", "name": "五粮液", "group": "重点关注"},
        {"code": "300750", "name": "宁德时代", "group": "短线操作"},
        {"code": "600036", "name": "招商银行", "group": "中长线"},
        {"code": "000333", "name": "美的集团", "group": "中长线"},
        {"code": "002415", "name": "海康威视", "group": "观察列表"},
        {"code": "601318", "name": "中国平安", "group": "观察列表"},
    ]
    pool_manager.import_from_list(samples)
    print(pool_manager.get_pool_summary())
